package com.technobium;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.StopwordAnalyzerBase;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

/**
 * TFIDF (term frequency / document frequency) - for use on small *mwk files
 */
// This was un-abstracted so that we can try and find phrases for the clustering
// code.
public class MahoutTermClusterMwkSnpt {

    // ---- Static

    private static final Logger LOG = LoggerFactory.getLogger(MahoutTermClusterMwkSnpt.class);
    private static final String BASE_PATH = System.getProperty("user.dir");
    private static final String POINTS_PATH = BASE_PATH + "/points";
    private static final String CLUSTERS_PATH = BASE_PATH + "/clusters";
    private static final String OUTPUT_PATH = BASE_PATH + "/output";

    public static void main(final String[] args) {
System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.main() - ");
        try {
            start();
        } catch (final Exception e) {
            LOG.error("MahoutTryIt failed", e);
        }
    }

    // ---- Fields

    private static final double[][] points = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 8, 8 }, { 9, 8 },
            { 8, 9 }, { 9, 9 } };

    private static final int numberOfClusters = 2;

    // ---- Methods

    private static void start() throws Exception {
System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.start() - ");
        final Configuration configuration = new Configuration();

        // Create input directories for data
        final File pointsDir = new File(POINTS_PATH);
        if (!pointsDir.exists()) {
            pointsDir.mkdir();
        }

        // read the point values and generate vectors from input data
        final List<Vector> vectors = vectorize(points);

        // Write data to sequence hadoop sequence files
        writePointsToFile(configuration, vectors);

        // Write initial centers for clusters
        writeClusterInitialCenters(configuration, vectors);

        // Run K-means algorithm
        final Path inputPath = new Path(POINTS_PATH);
        final Path clustersPath = new Path(CLUSTERS_PATH);
        final Path outputPath = new Path(OUTPUT_PATH);
        HadoopUtil.delete(configuration, outputPath);

        KMeansDriver.run(configuration, inputPath, clustersPath, outputPath, 0.001, 10, true, 0, false);

        // Read and print output values
        readAndPrintOutputValues(configuration);
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.start() - end");
    }

    private static void writePointsToFile(final Configuration configuration, final List<Vector> points)
            throws IOException {
System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - begin");
        final Path path = new Path(POINTS_PATH + "/pointsFile");
        FileSystem fs = FileSystem.getLocal(configuration);
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 1");
        final SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, path, IntWritable.class,
                VectorWritable.class);
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 2");

        int recNum = 0;
        final VectorWritable vec = new VectorWritable();

        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 3");
        for (final Vector point : points) {
            System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - point = " + point);
            vec.set(point);
            writer.append(new IntWritable(recNum++), vec);
        }
System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - end");
        writer.close();
    }

    private static void writeClusterInitialCenters(final Configuration configuration, final List<Vector> points)
            throws IOException {
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writeClusterInitialCenters() - ");
        final Path writerPath = new Path(CLUSTERS_PATH + "/part-00000");

        FileSystem fs = FileSystem.getLocal(configuration);
        final Path path = new Path(POINTS_PATH + "/pointsFile");
        final SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, writerPath, Text.class,
                Kluster.class);

        for (int i = 0; i < numberOfClusters; i++) {
            final Vector vec = points.get(i);

            // write the initial centers
            final Kluster cluster = new Kluster(vec, i, new EuclideanDistanceMeasure());
            writer.append(new Text(cluster.getIdentifier()), cluster);
        }

        writer.close();
    }

    private static void readAndPrintOutputValues(final Configuration configuration) throws IOException {
        final Path input = new Path(OUTPUT_PATH + "/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000");

        FileSystem fs = FileSystem.getLocal(configuration);
        final SequenceFile.Reader reader = new SequenceFile.Reader(fs, input, configuration);

        final IntWritable key = new IntWritable();
        final WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
        int count = 0;
        while (reader.next(key, value)) {
            LOG.info("{} belongs to cluster {}", value.toString(), key.toString());
        }
        reader.close();
        if (count == 0) {
            throw new RuntimeException("No  output pairs");
        }
    }

    // Read the points to vector from 2D array
    private static List<Vector> vectorize(final double[][] raw) {
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.vectorize() - ");
        final List<Vector> points = new ArrayList<Vector>();

        for (int i = 0; i < raw.length; i++) {
            final Vector vec = new RandomAccessSparseVector(raw[i].length);
            vec.assign(raw[i]);
            points.add(vec);
        }

        return points;
    }

    private static final int THRESHOLD = 1;

    // The biggest problem with this API is that there is a lot of I/O
    private static void main1(String args[]) throws Exception {

        System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog");

        Configuration configuration = new Configuration();
        String outputFolder = "temp_intermediate/";

        Path documentsSequencePath = new Path(outputFolder, "sequence");
        System.err.println(
                "SRIDHAR MahoutTermFinderMwkSnpt.main() - Creating sequence file from mwk snippet files, outputting files to sequence file "
                        + documentsSequencePath + " (large)");
        {
            SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(configuration), configuration,
                    documentsSequencePath, Text.class, Text.class);

            for (String path : new String[] { System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/aspergers",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/atletico",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/business",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/career",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/equalizer",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/productivity",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/self",
                    System.getProperty("user.home")
                            + "/sarnobat.git/mwk/snippets/self/approval_attention_social_status",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/self/cliquology_and_bullying/",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/soccer",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/tech/programming_tips",
                    System.getProperty("user.home")
                            + "/sarnobat.git/mwk/snippets/tech/programming_tips/functional_programming",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/travel",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/video_editing",
                    System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/wrestling", }) {
                Text id = new Text(Paths.get(path).getFileName().toString());
                DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(Paths.get(path));
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + id + "::" + Paths.get(path).toFile());
                try {
                    for (java.nio.file.Path fileInPath : stream) {
                        if (Files.isDirectory(fileInPath)) {
                            // listFiles(entry);
                        } else {
                            if (fileInPath.toFile().exists()) {
                                writer.append(id,
                                        new Text(FileUtils.readFileToString(Paths.get(fileInPath.toUri()).toFile())));
                            }
                        }
                    }
                } catch (IOException e3) {
                    throw e3;
                } finally {
                }
            }

            writer.close();
        }
        {
            Path tokenizedDocumentsPath = new Path(outputFolder, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
            System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Adding tokenized documents to folder "
                    + tokenizedDocumentsPath);
            System.err.println("MahoutTermFinder.calculateTfIdf() - Tokenzing documents, using "
                    + MyEnglishAnalyzer.class + " using reflection (yuck). Outputting to: " + tokenizedDocumentsPath);
            System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + documentsSequencePath + " ===> "
                    + tokenizedDocumentsPath);
            try {
                DocumentProcessor.tokenizeDocuments(documentsSequencePath, MyEnglishAnalyzer.class,
                        tokenizedDocumentsPath, configuration);
            } catch (Exception e) {
                e.printStackTrace();
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Could not instantiate "
                        + MyEnglishAnalyzer.class + ". Probably there is no public class and constructor.");
                return;
            }
            System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + tokenizedDocumentsPath + " ===> "
                    + new Path(outputFolder + "/" + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));
            DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(outputFolder),
                    DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration, 1, 1, 0.0f,
                    PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);
            // System.err.println("MahoutTermFinder.calculateTfIdf() - Creating term vectors
            // using input file " + new Path(outputFolder +
            // DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));
            Path tfidfPath = new Path(outputFolder + "tfidf");
            System.err.println("MahoutTermFinder.calculateTfIdf() - adding document frequencies to file " + tfidfPath);
            {
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - "
                        + new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER) + " ===> "
                        + tfidfPath);
                Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter.calculateDF(
                        new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER), tfidfPath,
                        configuration, 100);

                System.err.println("MahoutTermFinder.calculateTfIdf() - adding tfidf scores to file " + tfidfPath);
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - "
                        + new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER) + " ===> "
                        + tfidfPath);
                TFIDFConverter.processTfIdf(new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER),
                        tfidfPath, configuration, documentFrequencies, 1, 100, PartialVectorMerger.NO_NORMALIZING,
                        false, false, false, 1);
            }
        }
        System.err.println("MahoutTermFinder.main() - ??? ===> " + new Path(outputFolder, "dictionary.file-0"));
        System.err.println("MahoutTermFinder.main() - Reading dictionary into map. Dictionary of terms with IDs: "
                + new Path(outputFolder, "dictionary.file-0") + " (large)");
        Map<String, Object> dictionary;
        {
            // Create a vector numerical value for each term (e.g. "atletico" -> 4119)
            SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
                    new Path(outputFolder, "dictionary.file-0"), configuration);
            Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
            for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
                // System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - sequenceFile = "
                // + sequenceFile);
                // System.err.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
                termToOrdinalMappings2.put(sequenceFile.getFirst().toString(), sequenceFile.getSecond());
            }
            dictionary = termToOrdinalMappings2;
        }

        System.err.println("MahoutTermFinder.main() - Creating TFIDF Vectors");
        Map<String, Object> tfidf;
        {
            // Create a vector numerical value for each term (e.g. "atletico" -> 4119)
            SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
                    new Path(outputFolder, "tfidf/tfidf-vectors/part-r-00000"), configuration);
            Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
            for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
                // System.err.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
                termToOrdinalMappings2.put(sequenceFile.getFirst().toString(), sequenceFile.getSecond());
            }
            tfidf = termToOrdinalMappings2;
        }
        // System.err.println("MahoutTermFinder.main() - done");
        System.err.println("MahoutTermFinder.main() - Reading TFIDF Vectors (this will take a while)");
        Map<String, Map<String, Double>> ret1 = new HashMap<String, Map<String, Double>>();
        for (String file1 : tfidf.keySet()) {
            System.err.println("MahoutTermFinder.transform() " + file1);
            VectorWritable tfidf2 = (VectorWritable) tfidf.get(file1);
            Map<String, Double> transform;
            {
                // System.err.println("MahoutTermFinder.transform()");
                BiMap<String, Object> terms = HashBiMap.create();
                terms.putAll(dictionary);
                Map<Object, String> terms2 = terms.inverse();
                Map<Integer, String> m = new HashMap<Integer, String>();
                // System.err.println("MahoutTermFinder.convert() " + terms1.size());
                int i = 0;
                for (Object o : terms2.keySet()) {
                    String value = terms2.get(o);
                    // System.out.print(".");
                    if (value == null) {
                    }
                    Preconditions.checkNotNull(value, "Couldn't get value. o = " + o + ", terms1 = " + terms2);
                    if (i % 1000 == 0) {
                        // System.err.println("MahoutTermFinder.convert() " + file1 + " term " + i);
                        System.err.println("MahoutTermFinder.convert() " + file1 + " " + i + " "
                                + ((IntWritable) o).get() + "::" + value + " (term_id, term)");
                    }
                    m.put(((IntWritable) o).get(), value);
                    ++i;
                }
                Map<Integer, String> terms1 = m;
                Map<String, Double> ret = new HashMap<String, Double>();
                int j = 0;
                for (Element scoreElement : tfidf2.get().all()) {
                    int id = scoreElement.index();
                    if (!terms1.containsKey(id)) {
                        throw new RuntimeException("Couldn't find key " + id + ", only found " + terms1.keySet());
                    }
                    String term = (String) terms1.get(id);
                    double score = scoreElement.get();
                    if (j % 1000 == 0) {
                        // System.err.println("MahoutTermFinder.convert() " + file1 + " score element "
                        // + j);
                        System.err.println("MahoutTermFinder.convert() " + file1 + " " + j + " " + id + "::" + score
                                + " (term_id, score)");
                    }
                    ret.put(term, score);
                    // System.err.println("MahoutTermFinder.transform() term = " + term);
                    j++;
                }
                transform = ret;
            }
            ret1.put(file1, transform);
        }

        Map<String, Map<String, Double>> scores = ret1;
        System.err.println("MahoutTermFinderMwkSnpt.filter() - Discarding terms with low scores.");
        Map<String, Map<String, Double>> fileToHighScoreTermsToScores = new HashMap<String, Map<String, Double>>();
        for (String file : scores.keySet()) {
            Map<String, Double> tfidf1 = scores.get(file);
            Map<String, Double> ret = new HashMap<String, Double>();
            for (String s : tfidf1.keySet()) {
                if (tfidf1.get(s) > THRESHOLD) {
                    ret.put(s, tfidf1.get(s));
                }
            }
            Map<String, Double> filter2 = ret;
            // System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.filter() - " + file + "
            // --> " + filter2.size() + "(of " +tfidf.size()+")");
            fileToHighScoreTermsToScores.put(file, filter2);
            if (filter2.size() < 1) {
                throw new RuntimeException("No terms found for " + file);
            }
        }
        Map<String, Map<String, Double>> filter = fileToHighScoreTermsToScores;
        System.err.println("MahoutTermFinderMwkSnpt.main() - printing terms that satisfy the minimum score.");
        for (String filename : filter.keySet()) {
            // System.err.println("MahoutTermFinder.main()");
            Map<String, Double> scoresForDocument = filter.get(filename);
            List<Entry<String, Double>> sortedEntries = new ArrayList<Entry<String, Double>>(
                    scoresForDocument.entrySet()).subList(0, Math.min(20, scoresForDocument.size()));

            Collections.sort(sortedEntries, new Comparator<Entry<String, Double>>() {
                // @Override
                public int compare(Entry<String, Double> e1, Entry<String, Double> e2) {
                    return e1.getValue().compareTo(e2.getValue());
                }
            });
            for (Entry<String, Double> e : sortedEntries) {
                Integer number = (int) (e.getValue() * 10);
                String s = StringUtils.leftPad(number.toString(), 3);
                System.out.println(filename + ": " + s + " " + e.getKey());
            }
        }
    }

    public static class MyEnglishAnalyzer extends StopwordAnalyzerBase {
        private final CharArraySet stemExclusionSet;

        private static class DefaultSetHolder {
            static final CharArraySet DEFAULT_STOP_SET = StandardAnalyzer.STOP_WORDS_SET;
        }

        public MyEnglishAnalyzer(Version matchVersion) {
            this(matchVersion, DefaultSetHolder.DEFAULT_STOP_SET);
        }

        public MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords) {
            this(matchVersion, stopwords, CharArraySet.EMPTY_SET);
        }

        public MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords, CharArraySet stemExclusionSet) {
            super(matchVersion, stopwords);
            this.stemExclusionSet = CharArraySet.unmodifiableSet(CharArraySet.copy(matchVersion, stemExclusionSet));
        }

        @Override
        protected TokenStreamComponents createComponents(String fieldName, Reader reader) {
            final Tokenizer source = new StandardTokenizer(matchVersion, reader);
            TokenStream result = new StandardFilter(matchVersion, source);
            // prior to this we get the classic behavior, standardfilter does it for
            // us.
            if (matchVersion.onOrAfter(Version.LUCENE_31))
                result = new EnglishPossessiveFilter(matchVersion, result);
            result = new LowerCaseFilter(matchVersion, result);
            CharArraySet stopwords2;
            try {
                stopwords2 = getStopWords(System.getProperty("user.home") + "/github/mahout/stopwords.txt");
            } catch (IOException e) {
                try {
                    result.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
                e.printStackTrace();
                throw new RuntimeException(e);
            }
            result = new StopFilter(matchVersion, result, stopwords2);
            if (!stemExclusionSet.isEmpty())
                result = new SetKeywordMarkerFilter(result, stemExclusionSet);
            result = new PorterStemFilter(result);
            return new TokenStreamComponents(source, result);
        }

        private static CharArraySet getStopWords(String stoplist) throws IOException {
            List<String> ss = FileUtils.readLines(Paths.get(stoplist).toFile());
            CharArraySet ret = new CharArraySet(Version.LUCENE_CURRENT, ss, false);
            ret.addAll(ss);
            return ret;
        }
    }
}