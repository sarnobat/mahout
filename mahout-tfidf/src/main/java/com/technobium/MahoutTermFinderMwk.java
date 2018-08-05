package com.technobium;

import java.io.IOException;
import java.io.Reader;
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
 * TFIDF (term frequency / document frequency) - for regular mwk files
 */
public class MahoutTermFinderMwk {

  private static final int threshold = 7;

  private static final String STOPLIST = System.getProperty("user.home") + "/github/mahout/stopwords.txt";
  private static final String[] files = { System.getProperty("user.home") + "/sarnobat.git/mwk/technology.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/technology-linux.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/health.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/finance.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/geography.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/entertainment.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/soccer.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/people.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/productivity.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/atletico_madrid.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/atletico_documentary.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/atletico_articles_english.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/atletico_season_reviews.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/learning.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/design.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/girls.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/business.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/career.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/self.mwk",
      System.getProperty("user.home") + "/sarnobat.git/mwk/programming-tips.mwk" };

  static {
    System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog");
  }

  public static void main(String args[]) throws Exception {

    Configuration configuration = new Configuration();
    String outputFolder = "output/";
    Path documentsSequencePath = new Path(outputFolder, "sequence");
    selectDocuments(FileSystem.get(configuration), configuration, documentsSequencePath, files);
    calculateTfIdf(documentsSequencePath, configuration, new Path(outputFolder + "tfidf"), outputFolder,
        new Path(outputFolder, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER),
        new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));

    System.out.println("MahoutTermFinder.main() - Creating dictionary");
    Map<String, Object> dictionary = sequenceFileToMap(new Path(outputFolder, "dictionary.file-0"), configuration);

    System.out.println("MahoutTermFinder.main() - Creating TFIDF Vectors (this will take a while)");
    Map<String, Object> tfidf = sequenceFileToMap(new Path(outputFolder, "tfidf/tfidf-vectors/part-r-00000"),
        configuration);
    System.out.println("MahoutTermFinder.main() - done");

    Map<String, Map<String, Double>> scores = transform(tfidf, dictionary);
    Map<String, Map<String, Double>> filter = filter(scores);
    for (String filename : filter.keySet()) {
      Map<String, Double> scoresForDocument = filter.get(filename);
      List<Entry<String, Double>> sortedEntries = new ArrayList<Entry<String, Double>>(scoresForDocument.entrySet());

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

  private static Map<String, Map<String, Double>> filter(Map<String, Map<String, Double>> scores) {
    Map<String, Map<String, Double>> ret = new HashMap<String, Map<String, Double>>();
    for (String file : scores.keySet()) {
      Map<String, Double> tfidf = scores.get(file);
      ret.put(file, filter2(tfidf));
    }
    return ret;
  }

  private static Map<String, Double> filter2(Map<String, Double> tfidf) {
    Map<String, Double> ret = new HashMap<String, Double>();
    for (String s : tfidf.keySet()) {
      if (tfidf.get(s) > threshold) {
        ret.put(s, tfidf.get(s));
      }
    }
    return ret;
  }

  private static Map<String, Map<String, Double>> transform(Map<String, Object> tfidfs,
      Map<String, Object> dictionary) {
    Map<String, Map<String, Double>> ret = new HashMap<String, Map<String, Double>>();
    for (String file : tfidfs.keySet()) {
      VectorWritable tfidf = (VectorWritable) tfidfs.get(file);
      ret.put(file, transform(tfidf, dictionary));
    }
    return ret;
  }

  private static Map<String, Double> transform(VectorWritable tfidf, Map<String, Object> dictionary) {
    BiMap<String, Object> terms = HashBiMap.create();
    terms.putAll(dictionary);
    Map<Object, String> terms2 = terms.inverse();
    Map<Integer, String> terms1 = convert(terms2);
    Map<String, Double> ret = new HashMap<String, Double>();
    for (Element e : tfidf.get().all()) {
      double score = e.get();
      int id = e.index();
      if (!terms1.containsKey(id)) {
        throw new RuntimeException("Couldn't find key " + id + ", only found " + terms1.keySet());
      }
      String term = (String) terms1.get(id);
      ret.put(term, score);
    }
    return ret;
  }

  private static Map<Integer, String> convert(Map<Object, String> terms1) {
    Map<Integer, String> m = new HashMap<Integer, String>();
    for (Object o : terms1.keySet()) {
      String value = terms1.get(o);
      if (value == null) {
      }
      Preconditions.checkNotNull(value, "Couldn't get value. o = " + o + ", terms1 = " + terms1);
      m.put(((IntWritable) o).get(), value);
    }
    return m;
  }

  static void selectDocuments(FileSystem fileSystem, Configuration configuration, Path documentsSequencePath,
      String[] files) throws IOException {

    SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem, configuration, documentsSequencePath, Text.class,
        Text.class);

    for (String path : files) {
      Text id = new Text(Paths.get(path).getFileName().toString());
      Text text = new Text(FileUtils.readFileToString(Paths.get(path).toFile()));
      writer.append(id, text);
    }

    writer.close();
  }

  static void calculateTfIdf(Path documentsSequencePath, Configuration configuration, Path tfidfPath,
      String outputFolder, Path tokenizedDocumentsPath, Path termFrequencyVectorsPath)
      throws ClassNotFoundException, IOException, InterruptedException {

    System.out.println("MahoutTermFinder.calculateTfIdf() - Tokenzing documents");
    DocumentProcessor.tokenizeDocuments(documentsSequencePath, MyEnglishAnalyzer.class, tokenizedDocumentsPath,
        configuration);
    System.out.println("MahoutTermFinder.calculateTfIdf() - Creating term vectors");
    DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(outputFolder),
        DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration, 1, 1, 0.0f,
        PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);
    System.out.println("MahoutTermFinder.calculateTfIdf() - Creating document frequencies");
    Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter.calculateDF(termFrequencyVectorsPath, tfidfPath,
        configuration, 100);
    System.out.println("MahoutTermFinder.calculateTfIdf() - creating tfidf scores");
    TFIDFConverter.processTfIdf(termFrequencyVectorsPath, tfidfPath, configuration, documentFrequencies, 1, 100,
        PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
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

  static void printSequenceFile(Path path, Configuration configuration) {
    Configuration configuration2 = configuration;
    SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(path,
        configuration2);
    for (Pair<Writable, Writable> pair : iterable) {
      System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
    }
  }

  static Map<String, Object> sequenceFileToMap(Path path, Configuration configuration) {
    SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(path,
        configuration);
    Map<String, Object> m = new HashMap<String, Object>();
    for (Pair<Writable, Writable> pair : iterable) {
      // System.out.format("%10s -> %s\n", pair.getFirst(),
      // pair.getSecond());
      m.put(pair.getFirst().toString(), pair.getSecond());
    }
    return m;
  }

  public static class MyEnglishAnalyzer extends StopwordAnalyzerBase {
    private final CharArraySet stemExclusionSet;

    private static CharArraySet getDefaultStopSet() {
      return DefaultSetHolder.DEFAULT_STOP_SET;
    }

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
        stopwords2 = getStopWords(STOPLIST);
      } catch (IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
      result = new StopFilter(matchVersion, result, stopwords2);
      if (!stemExclusionSet.isEmpty())
        result = new SetKeywordMarkerFilter(result, stemExclusionSet);
      result = new PorterStemFilter(result);
      return new TokenStreamComponents(source, result);
    }

    private CharArraySet getStopWords(String stoplist) throws IOException {
      List<String> ss = FileUtils.readLines(Paths.get(stoplist).toFile());
      CharArraySet ret = new CharArraySet(Version.LUCENE_CURRENT, ss, false);
      ret.addAll(ss);
      return ret;
    }
  }
}