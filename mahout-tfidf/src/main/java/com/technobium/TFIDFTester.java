package com.technobium;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

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
		printSequenceFile(new Path(outputFolder, "dictionary.file-0"), configuration);
		printSequenceFile(new Path(outputFolder + "tf-vectors/part-r-00000"), configuration);
		printSequenceFile(new Path(outputFolder + "tfidf/df-count/part-r-00000"), configuration);
		printSequenceFile(new Path(outputFolder + "tfidf/tfidf-vectors/part-r-00000"),
				configuration);

	}

	static void createTestDocuments(FileSystem fileSystem, Configuration configuration,
			Path documentsSequencePath) throws IOException {
		Path documentsSequencePath2 = documentsSequencePath;
		SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem, configuration,
				documentsSequencePath2, Text.class, Text.class);

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
}