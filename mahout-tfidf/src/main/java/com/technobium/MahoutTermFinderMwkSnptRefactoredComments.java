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
// This was un-abstracted so that we can try and find phrases for the clustering code.
// This doesn't fail, but the documents are wrong.
public class MahoutTermFinderMwkSnptRefactoredComments {

	private static final int THRESHOLD = 1;

	// The biggest problem with this API is that there is a lot of I/O
	public static void main(String args[]) throws Exception {

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
			String[] dirs = new String[] { System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/aspergers",
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
					System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/wrestling" };
			// ----------------------------------------------------------------------
			// 1) Write [doc path, doc content] pairs to a concurrent map
			// ----------------------------------------------------------------------

			for (String path : dirs) {
				DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(Paths.get(path));
				try {
					for (java.nio.file.Path fileInPath : stream) {
						if (Files.isDirectory(fileInPath)) {
							// listFiles(entry);
						} else {
							if (fileInPath.toFile().exists()) {
								Text id = new Text(fileInPath.getFileName().toString());
								String readFileToString = FileUtils
										.readFileToString(Paths.get(fileInPath.toUri()).toFile());
								System.err.println("SRIDHAR - readFileToString.length() " + readFileToString.length());
								System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + id + "::"
										+ readFileToString.substring(0, Math.min(readFileToString.length() - 1, 30)));
								// This is wrong, the id is the parent dir, not the file
								writer.append(id, new Text(readFileToString));
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
		// ----------------------------------------------------------------------
		// 2) Tokenizing documents
		// ----------------------------------------------------------------------
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
			} catch (IllegalStateException e) {
				e.printStackTrace();
				System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Could not instantiate "
						+ MyEnglishAnalyzer.class + ". Probably there is no public class and constructor.");
				return;
			}
			System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + tokenizedDocumentsPath + " ===> "
					+ new Path(outputFolder + "/" + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));
			// ----------------------------------------------------------------------
			// 2) Counting term frequencies
			// ----------------------------------------------------------------------
			DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(outputFolder),
					DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration, 1, 1, 0.0f,
					PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);
//            System.err.println("MahoutTermFinder.calculateTfIdf() - Creating term vectors using input file " + new Path(outputFolder + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));
			// ----------------------------------------------------------------------
			// 4) Counting document frequencies
			// ----------------------------------------------------------------------
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

		// ----------------------------------------------------------------------
		// 5) Clustering
		// ----------------------------------------------------------------------
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
