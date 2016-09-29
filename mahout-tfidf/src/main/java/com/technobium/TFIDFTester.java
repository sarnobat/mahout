package com.technobium;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
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
import com.google.common.collect.Maps;

public class TFIDFTester {

	public static void main(String args[]) throws Exception {

		Configuration configuration = new Configuration();
		String outputFolder = "output/";
		Path documentsSequencePath = new Path(outputFolder, "sequence");
		createTestDocuments(FileSystem.get(configuration), configuration, documentsSequencePath);

		calculateTfIdf(documentsSequencePath, configuration, new Path(outputFolder + "tfidf"),
				outputFolder, new Path(outputFolder,
						DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER), new Path(outputFolder
						+ DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));

		printSequenceFile(documentsSequencePath, configuration);
		printSequenceFile(new Path(outputFolder + "wordcount/part-r-00000"), configuration);

		System.out.println("Dictionary File");
		Map<String, Object> dictionary = sequenceFileToMap(new Path(outputFolder,
				"dictionary.file-0"), configuration);
		System.out.println("TFIDFTester.main() - " + dictionary);
		printSequenceFile(new Path(outputFolder, "dictionary.file-0"), configuration);
		printSequenceFile(new Path(outputFolder + "tf-vectors/part-r-00000"), configuration);
		printSequenceFile(new Path(outputFolder + "tfidf/df-count/part-r-00000"), configuration);

		System.out.println("TFIDF Vectors");
		printSequenceFile(new Path(outputFolder + "tfidf/tfidf-vectors/part-r-00000"),
				configuration);
		Map<String, Object> tfidf = sequenceFileToMap(new Path(outputFolder,
				"tfidf/tfidf-vectors/part-r-00000"), configuration);
		System.out.println("TFIDFTester.main() - " + tfidf);

		Map<String, Map<String, Double>> scores = transform(tfidf, dictionary);
		System.out.println(scores);

	}

	private static Map<String, Map<String, Double>> transform(Map<String, Object> tfidfs,
			Map<String, Object> dictionary) {
		Map<String, Map<String, Double>> ret = new HashMap<String, Map<String, Double>>();
		for (String file : tfidfs.keySet()) {
			System.out.println("TFIDFTester.transform() file = " + file);
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
		System.out.println("TFIDFTester.transform() " + terms2.keySet());
		Map<Integer, String> terms1 = convert(terms2);
		System.out.println("TFIDFTester.transform() terms1 = " + terms1);
		System.out.println("TFIDFTester.transform() " + terms1.keySet());
		Map<String, Double> ret = new HashMap<String, Double>();
		for (Element e : tfidf.get().all()) {
			double score = e.get();
			int id = e.index();
			System.out.println("TFIDFTester.transform() - id, score: " + id + " = " + score);
			if (!terms1.containsKey(id)) {
				throw new RuntimeException("Couldn't find key " + id + ", only found "
						+ terms1.keySet());
			}
			String term = (String) terms1.get(id);
			System.out.println("TFIDFTester.transform() term = " + term);
			ret.put(term, score);
		}
		System.out.println("TFIDFTester.transform() ret = " + ret);
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

	static void createTestDocuments(FileSystem fileSystem, Configuration configuration,
			Path documentsSequencePath) throws IOException {

		SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem, configuration,
				documentsSequencePath, Text.class, Text.class);

		Text id1 = new Text("Document 1");
		Text text1 = new Text("I saw a yellow car and a green car.");
		writer.append(id1, text1);

		Text id2 = new Text("Document 2");
		Text text2 = new Text("You saw a red car.");
		writer.append(id2, text2);

		writer.close();
	}

	static void calculateTfIdf(Path documentsSequencePath, Configuration configuration,
			Path tfidfPath, String outputFolder, Path tokenizedDocumentsPath,
			Path termFrequencyVectorsPath) throws ClassNotFoundException, IOException,
			InterruptedException {

		// Tokenize the documents using Apache Lucene StandardAnalyzer
		DocumentProcessor.tokenizeDocuments(documentsSequencePath, StandardAnalyzer.class,
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
			System.out.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
			System.out.println("TFIDFTester.sequenceFileToMap() - " + pair.getSecond().getClass());
			m.put(pair.getFirst().toString(), pair.getSecond());
		}
		return m;
	}
}