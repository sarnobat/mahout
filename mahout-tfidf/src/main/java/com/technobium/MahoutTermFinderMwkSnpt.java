package com.technobium;

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
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

/**
 * TFIDF (term frequency / document frequency) - for use on small *mwk files
 */
public class MahoutTermFinderMwkSnpt {

    private static final int THRESHOLD = 1;

    public static void main(String args[]) throws Exception {

        System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog");

        Configuration configuration = new Configuration();
        String outputFolder = "temp_intermediate/";

        Path documentsSequencePath = new Path(outputFolder, "sequence");
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
                DirectoryStream<java.nio.file.Path> stream = null;
                try {
                    stream = Files.newDirectoryStream(Paths.get(path));
                    for (java.nio.file.Path fileInPath : stream) {
                        if (Files.isDirectory(fileInPath)) {
                            // listFiles(entry);
                        } else {
                            if (fileInPath.toFile().exists()) {
                                Text text = new Text(
                                        FileUtils.readFileToString(Paths.get(fileInPath.toUri()).toFile()));
                                writer.append(id, text);
                            }
                        }
                    }
                } catch (IOException e3) {
                    throw e3;
                } finally {
                    if (stream != null) {
                        stream.close();
                    }
                }
            }

            writer.close();
        }
        Path tfidfPath = new Path(outputFolder + "tfidf");
        Path tokenizedDocumentsPath = new Path(outputFolder, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
        Path termFrequencyVectorsPath = new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER);
        System.err.println("MahoutTermFinder.calculateTfIdf() - Tokenzing documents");
        DocumentProcessor.tokenizeDocuments(documentsSequencePath, MyEnglishAnalyzer.class, tokenizedDocumentsPath,
                configuration);
        System.err.println("MahoutTermFinder.calculateTfIdf() - Creating term vectors");
        DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(outputFolder),
                DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration, 1, 1, 0.0f,
                PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);
        System.err.println("MahoutTermFinder.calculateTfIdf() - Creating document frequencies");
        {
            Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter.calculateDF(termFrequencyVectorsPath,
                    tfidfPath, configuration, 100);
            System.err.println("MahoutTermFinder.calculateTfIdf() - creating tfidf scores");
            TFIDFConverter.processTfIdf(termFrequencyVectorsPath, tfidfPath, configuration, documentFrequencies, 1, 100,
                    PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
        }
        System.err.println("MahoutTermFinder.main() - Creating dictionary");
        Map<String, Object> dictionary;
        {
            // Create a vector numerical value for each term (e.g. "atletico" -> 4119)
            SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
                    new Path(outputFolder, "dictionary.file-0"), configuration);
            Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
            for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
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
            // System.err.println("MahoutTermFinder.transform() " + file);
            VectorWritable tfidf2 = (VectorWritable) tfidf.get(file1);
            ret1.put(file1, transform(tfidf2, dictionary));
        }

        Map<String, Map<String, Double>> scores = ret1;
        System.err.println("MahoutTermFinderMwkSnpt.filter() - ");
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
        System.err.println("MahoutTermFinderMwkSnpt.main() - printing terms");
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

    private static Map<String, Double> transform(VectorWritable tfidf, Map<String, Object> dictionary) {
        // System.err.println("MahoutTermFinder.transform()");
        BiMap<String, Object> terms = HashBiMap.create();
        terms.putAll(dictionary);
        Map<Object, String> terms2 = terms.inverse();
        Map<Integer, String> m = new HashMap<Integer, String>();
        // System.err.println("MahoutTermFinder.convert() " + terms1.size());
        int i = 0;
        for (Object o : terms2.keySet()) {
            if (i % 100 == 0) {
                // System.err.println("MahoutTermFinder.convert() " + i );
            }
            String value = terms2.get(o);
            // System.out.print(".");
            if (value == null) {
            }
            Preconditions.checkNotNull(value, "Couldn't get value. o = " + o + ", terms1 = " + terms2);
            m.put(((IntWritable) o).get(), value);
            ++i;
        }
        Map<Integer, String> terms1 = m;
        Map<String, Double> ret = new HashMap<String, Double>();
        for (Element e : tfidf.get().all()) {
            double score = e.get();
            int id = e.index();
            if (!terms1.containsKey(id)) {
                throw new RuntimeException("Couldn't find key " + id + ", only found " + terms1.keySet());
            }
            String term = (String) terms1.get(id);
            ret.put(term, score);
            // System.err.println("MahoutTermFinder.transform() term = " + term);
        }
        return ret;
    }

    // private static class SridharAnalyzer extends Analyzer {
    //
    // /* This is the only function that we need to override for our analyzer.
    // * It takes in a java.io.Reader object and saves the tokenizer and list
    // * of token filters that operate on it.
    // */
    // @Override
    // protected TokenStreamComponents createComponents(String arg0, Reader
    // arg1) {
    // Tokenizer tokenizer = new PlusSignTokenizer(reader);
    // TokenStream filter = new EmptyStringTokenFilter(tokenizer);
    // filter = new LowerCaseFilter(filter);
    // return new TokenStreamComponents(tokenizer, filter);
    // }
    // }

    // private static void printSequenceFile(Path path, Configuration configuration)
    // {
    // Configuration configuration2 = configuration;
    // SequenceFileIterable<Writable, Writable> iterable = new
    // SequenceFileIterable<Writable, Writable>(path,
    // configuration2);
    // for (Pair<Writable, Writable> pair : iterable) {
    // System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
    // }
    // }

    private static class MyEnglishAnalyzer extends StopwordAnalyzerBase {
        private final CharArraySet stemExclusionSet;

        // private static CharArraySet getDefaultStopSet() {
        // return DefaultSetHolder.DEFAULT_STOP_SET;
        // }

        private static class DefaultSetHolder {
            static final CharArraySet DEFAULT_STOP_SET = StandardAnalyzer.STOP_WORDS_SET;
        }

        private MyEnglishAnalyzer(Version matchVersion) {
            this(matchVersion, DefaultSetHolder.DEFAULT_STOP_SET);
        }

        private MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords) {
            this(matchVersion, stopwords, CharArraySet.EMPTY_SET);
        }

        private MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords, CharArraySet stemExclusionSet) {
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