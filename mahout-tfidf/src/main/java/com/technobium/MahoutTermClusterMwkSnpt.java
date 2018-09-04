package com.technobium;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.Analyzer.TokenStreamComponents;
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
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

/**
 * I'm not sure why we're trying to cluster the mwk files (as opposed to mwk
 * snippets). Each mwk file is suppsoed to be different.
 */
public class MahoutTermClusterMwkSnpt {

    public static void main(String args[]) throws Exception {

        String outputFolder = "output/";
        Configuration configuration = new Configuration();
        FileSystem fileSystem = FileSystem.get(configuration);
        Path documentsSequencePath = new Path(outputFolder, "sequence");
        Path tokenizedDocumentsPath = new Path(outputFolder, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
        Path tfidfPath = new Path(outputFolder + "tfidf");
        Path termFrequencyVectorsPath = new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER);
        {
            SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem, configuration, documentsSequencePath,
                    Text.class, Text.class);

            {

                System.err.println(
                        "SRIDHAR MahoutTermFinderMwkSnpt.main() - Creating sequence file from mwk snippet files, outputting files to sequence file "
                                + documentsSequencePath + " (large)");

                int max = 0;
                boolean maxReached = false;
                for (String path : new String[] {
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/aspergers",
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
                    Text cateogoryDir = new Text(Paths.get(path).getFileName().toString());
                    DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(Paths.get(path));
                    System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + cateogoryDir + "::"
                            + Paths.get(path).toFile());
                    try {
                        for (java.nio.file.Path fileInPath : stream) {
                            System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + fileInPath);
                            max++;
                            if (max > 100) {
                                maxReached = true;
                                break;
                            }
                            if (Files.isDirectory(fileInPath)) {
                                // listFiles(entry);
                                System.out.println(
                                        "SRIDHAR MahoutTermClusterMwkSnpt.writeToSequenceFile() - skipping nested dir: "
                                                + fileInPath);
                            } else {
                                if (fileInPath.toFile().exists()) {
                                    writer.append(new Text(fileInPath.toAbsolutePath().toString()), new Text(
                                            FileUtils.readFileToString(Paths.get(fileInPath.toUri()).toFile())));
                                }
                            }
                        }
                        if (maxReached) {
                            break;
                        }
                    } catch (IOException e3) {
                        throw e3;
                    } finally {
                    }
                }
            }
            if (false) {
                Text id1 = new Text("Document 1");
                Text text1 = new Text("John saw a red car.");
                writer.append(id1, text1);

                Text id2 = new Text("Document 2");
                Text text2 = new Text("Marta found a red bike.");
                writer.append(id2, text2);

                Text id3 = new Text("Document 3");
                Text text3 = new Text("Don need a blue coat.");
                writer.append(id3, text3);

                Text id4 = new Text("Document 4");
                Text text4 = new Text("Mike bought a blue boat.");
                writer.append(id4, text4);

                Text id5 = new Text("Document 5");
                Text text5 = new Text("Albert wants a blue dish.");
                writer.append(id5, text5);

                Text id6 = new Text("Document 6");
                Text text6 = new Text("Lara likes blue glasses.");
                writer.append(id6, text6);

                Text id7 = new Text("Document 7");
                Text text7 = new Text("Donna, do you have red apples?");
                writer.append(id7, text7);

                Text id8 = new Text("Document 8");
                Text text8 = new Text("Sonia needs blue books.");
                writer.append(id8, text8);

                Text id9 = new Text("Document 9");
                Text text9 = new Text("I like blue eyes.");
                writer.append(id9, text9);

                Text id10 = new Text("Document 10");
                Text text10 = new Text("Arleen has a red carpet.");
                writer.append(id10, text10);
                writer.close();
            }
            writer.close();
        }

        {
            DocumentProcessor.tokenizeDocuments(documentsSequencePath, MyEnglishAnalyzer.class, tokenizedDocumentsPath,
                    configuration);

            DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(outputFolder),
                    DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration, 1, 1, 0.0f,
                    PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);

            Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter.calculateDF(termFrequencyVectorsPath,
                    tfidfPath, configuration, 100);

            TFIDFConverter.processTfIdf(termFrequencyVectorsPath, tfidfPath, configuration, documentFrequencies, 1, 100,
                    PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
        }
        {
            String vectorsFolder = outputFolder + "tfidf/tfidf-vectors/";
            String canopyCentroids = outputFolder + "canopy-centroids";
            String clusterOutput = outputFolder + "clusters";

            FileSystem fs = FileSystem.get(configuration);
            Path oldClusterPath = new Path(clusterOutput);

            if (fs.exists(oldClusterPath)) {
                fs.delete(oldClusterPath, true);
            }

            // original
            if (false) {
                CanopyDriver.run(new Path(vectorsFolder), new Path(canopyCentroids), new EuclideanDistanceMeasure(), 20,
                        5, true, 0, true);

                FuzzyKMeansDriver.run(new Path(vectorsFolder), new Path(canopyCentroids, "clusters-0-final"),
                        new Path(clusterOutput), 0.01, 20, 2, true, true, 0, false);
            } else {
                CanopyDriver.run(new Path(vectorsFolder), new Path(canopyCentroids), new TanimotoDistanceMeasure(), 20,
                        5, true, 0, true);

                FuzzyKMeansDriver.run(new Path(vectorsFolder), new Path(canopyCentroids, "clusters-0-final"),
                        new Path(clusterOutput), 0.01, 20, 2, true, true, 0, false);

            }
        }
        {
            SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(
                    documentsSequencePath, configuration);
            for (Pair<Writable, Writable> pair : iterable) {
                // System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());

            }
        }
        System.out.println("\n Clusters: ");
        {
            SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(
                    new Path(outputFolder + "clusters/clusteredPoints/part-m-00000"), configuration);
            for (Pair<Writable, Writable> pair : iterable) {
                java.nio.file.Path path = Paths
                        .get(((NamedVector) ((WeightedPropertyVectorWritable) pair.getSecond()).getVector()).getName());
                System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond().toString().substring(0, 20) + " :: "
                        + path.getFileName() + "::" + getContent(path));
            }
        }
    }

    private static String getContent(java.nio.file.Path path) throws IOException {
        String content = "";
        List<String> readAllLines = Files.readAllLines(path);
        for (String l : readAllLines) {
            content += l;
        }
        String substring = content.length() < 50 ? content.substring(0, content.length()) : content.substring(0, 50);
        return substring;
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
