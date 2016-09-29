package com.technobium;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
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

public class TFIDFTester {

	private static final int threshold = 7;

	public static void main(String args[]) throws Exception {

		Configuration configuration = new Configuration();
		String outputFolder = "output/";
		Path documentsSequencePath = new Path(outputFolder, "sequence");
		createTestDocuments(FileSystem.get(configuration), configuration, documentsSequencePath);

		calculateTfIdf(documentsSequencePath, configuration, new Path(outputFolder + "tfidf"),
				outputFolder, new Path(outputFolder,
						DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER), new Path(outputFolder
						+ DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));

		// printSequenceFile(documentsSequencePath, configuration);
		// printSequenceFile(new Path(outputFolder + "wordcount/part-r-00000"),
		// configuration);

		System.out.println("Dictionary File");
		Map<String, Object> dictionary = sequenceFileToMap(new Path(outputFolder,
				"dictionary.file-0"), configuration);
		// System.out.println("TFIDFTester.main() - " + dictionary);
		// printSequenceFile(new Path(outputFolder, "dictionary.file-0"),
		// configuration);
		// printSequenceFile(new Path(outputFolder + "tf-vectors/part-r-00000"),
		// configuration);
		// printSequenceFile(new Path(outputFolder +
		// "tfidf/df-count/part-r-00000"), configuration);

		System.out.println("TFIDF Vectors");
		// printSequenceFile(new Path(outputFolder +
		// "tfidf/tfidf-vectors/part-r-00000"),
		// configuration);
		Map<String, Object> tfidf = sequenceFileToMap(new Path(outputFolder,
				"tfidf/tfidf-vectors/part-r-00000"), configuration);
//		System.out.println("TFIDFTester.main() - " + tfidf);

		Map<String, Map<String, Double>> scores = transform(tfidf, dictionary);
		Map<String, Map<String, Double>> filter = filter(scores);
		for (String filename : filter.keySet()) {
			System.out.println(filename);
			System.out.println(filter.get(filename));
			System.out.println();
		}
		// System.out.println(filter);

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

	private static Map<String, Double> transform(VectorWritable tfidf,
			Map<String, Object> dictionary) {
		BiMap<String, Object> terms = HashBiMap.create();
		terms.putAll(dictionary);
		Map<Object, String> terms2 = terms.inverse();
		Map<Integer, String> terms1 = convert(terms2);
		Map<String, Double> ret = new HashMap<String, Double>();
		for (Element e : tfidf.get().all()) {
			double score = e.get();
			int id = e.index();
			if (!terms1.containsKey(id)) {
				throw new RuntimeException("Couldn't find key " + id + ", only found "
						+ terms1.keySet());
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
			Preconditions.checkNotNull(value, "Couldn't get value. o = " + o + ", terms1 = "
					+ terms1);
			m.put(((IntWritable) o).get(), value);
		}
		return m;
	}

	static void createTestDocuments(FileSystem fileSystem, Configuration configuration,
			Path documentsSequencePath) throws IOException {

		SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem, configuration,
				documentsSequencePath, Text.class, Text.class);
		
		String[] files = {
				System.getProperty("user.home") + "/sarnobat.git/mwk/technology.mwk",
				System.getProperty("user.home") + "/sarnobat.git/mwk/technology-linux.mwk",
				System.getProperty("user.home") + "/sarnobat.git/mwk/health.mwk",
				System.getProperty("user.home") + "/sarnobat.git/mwk/finance.mwk",
				System.getProperty("user.home") + "/sarnobat.git/mwk/geography.mwk",
				System.getProperty("user.home") + "/sarnobat.git/mwk/entertainment.mwk",
				
		};
		for (String path : files) {
			Text id = new Text(Paths.get(path).getFileName().toString());
			Text text = new Text(FileUtils.readFileToString(Paths.get(
					path).toFile()));
			writer.append(id, text);
		}

		Text id1 = new Text("learning.mwk");
		Text text1 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/learning.mwk").toFile()));
		writer.append(id1, text1);

		Text id2 = new Text("design.mwk");
		Text text2 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/design.mwk").toFile()));
		writer.append(id2, text2);

		Text id3 = new Text("girls.mwk");
		Text text3 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/girls.mwk").toFile()));
		writer.append(id3, text3);

		Text id4 = new Text("business.mwk");
		Text text4 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/business.mwk").toFile()));
		writer.append(id4, text4);

		Text id5 = new Text("career.mwk");
		Text text5 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/career.mwk").toFile()));
		writer.append(id5, text5);

		Text id6 = new Text("self.mwk");
		Text text6 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/self.mwk").toFile()));
		writer.append(id6, text6);
		
		Text id7 = new Text("programming-tips.mwk");
		Text text7 = new Text(FileUtils.readFileToString(Paths.get(
				System.getProperty("user.home") + "/sarnobat.git/mwk/programming-tips.mwk").toFile()));
		writer.append(id7, text7);

		writer.close();
	}

	static void calculateTfIdf(Path documentsSequencePath, Configuration configuration,
			Path tfidfPath, String outputFolder, Path tokenizedDocumentsPath,
			Path termFrequencyVectorsPath) throws ClassNotFoundException, IOException,
			InterruptedException {

		// Tokenize the documents using Apache Lucene StandardAnalyzer
		DocumentProcessor.tokenizeDocuments(documentsSequencePath, EnglishAnalyzer.class,
				tokenizedDocumentsPath, configuration);

		DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(
				outputFolder), DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration,
				1, 1, 0.0f, PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);

		Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter.calculateDF(
				termFrequencyVectorsPath, tfidfPath, configuration, 100);

		TFIDFConverter.processTfIdf(termFrequencyVectorsPath, tfidfPath, configuration,
				documentFrequencies, 1, 100, PartialVectorMerger.NO_NORMALIZING, false, false,
				false, 1);
	}

	static void printSequenceFile(Path path, Configuration configuration) {
		Configuration configuration2 = configuration;
		SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(
				path, configuration2);
		for (Pair<Writable, Writable> pair : iterable) {
			System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
		}
	}

	static Map<String, Object> sequenceFileToMap(Path path, Configuration configuration) {
		SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(
				path, configuration);
		Map<String, Object> m = new HashMap<String, Object>();
		for (Pair<Writable, Writable> pair : iterable) {
//			System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
			m.put(pair.getFirst().toString(), pair.getSecond());
		}
		return m;
	}
}