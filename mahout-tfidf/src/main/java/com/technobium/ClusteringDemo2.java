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
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

/**
 * This one works. Use it as a working baseline
 */
public class ClusteringDemo2 {

	private static String outputFolder;
	private static Configuration configuration;
	private static FileSystem fileSystem;
	private static Path documentsSequencePath;
	private static Path tokenizedDocumentsPath;
	private static Path tfidfPath;
	private static Path termFrequencyVectorsPath;

	public static void main(String args[]) throws Exception {

		configuration = new Configuration();
		fileSystem = FileSystem.get(configuration);

		outputFolder = "output/";
		documentsSequencePath = new Path(outputFolder, "sequence");
		tokenizedDocumentsPath = new Path(outputFolder,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
		tfidfPath = new Path(outputFolder + "tfidf");
		termFrequencyVectorsPath = new Path(outputFolder
				+ DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER);
		
		createTestDocuments();
		calculateTfIdf();
		clusterDocs();

		printSequenceFile(documentsSequencePath);

		System.out.println("\n Clusters: ");
		printSequenceFile(new Path(outputFolder
				+ "clusters/clusteredPoints/part-m-00000"));
	}

	private static  void createTestDocuments() throws IOException {
		SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem,
				configuration, documentsSequencePath, Text.class, Text.class);

		Text id1 = new Text("Document 1");
		Text text1 = new Text("Atletico Madrid win");
		writer.append(id1, text1);

		Text id6 = new Text("Document 6");
		Text text6 = new Text("Both apple and orange are fruit");
		writer.append(id6, text6);
		
		Text id7 = new Text("Document 7");
		Text text7 = new Text("Both orange and apple are fruit");
		writer.append(id7, text7);
		
		writer.close();
	}

	private static  void calculateTfIdf() throws ClassNotFoundException, IOException,
			InterruptedException {
		DocumentProcessor.tokenizeDocuments(documentsSequencePath,
				StandardAnalyzer.class, tokenizedDocumentsPath, configuration);

		DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath,
				new Path(outputFolder),
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
				configuration, 1, 1, 0.0f, PartialVectorMerger.NO_NORMALIZING,
				true, 1, 100, false, false);

		Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter
				.calculateDF(termFrequencyVectorsPath, tfidfPath,
						configuration, 100);

		TFIDFConverter.processTfIdf(termFrequencyVectorsPath, tfidfPath,
				configuration, documentFrequencies, 1, 100,
				PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
	}

	private static void clusterDocs() throws ClassNotFoundException, IOException,
			InterruptedException {
		String vectorsFolder = outputFolder + "tfidf/tfidf-vectors/";
		String canopyCentroids = outputFolder + "canopy-centroids";
		String clusterOutput = outputFolder + "clusters";

		FileSystem fs = FileSystem.get(configuration);
		Path oldClusterPath = new Path(clusterOutput);

		if (fs.exists(oldClusterPath)) {
			fs.delete(oldClusterPath, true);
		}
		{
			// CosineDistanceMeasure
			CanopyDriver.run(new Path(vectorsFolder),
					new Path(canopyCentroids), new CosineDistanceMeasure(),
					0.2, 0.2, true, 1, true);

			FuzzyKMeansDriver.run(new Path(vectorsFolder), new Path(
					canopyCentroids, "clusters-0-final"), new Path(
					clusterOutput), 0.01, 20, 2, true, true, 0, false);
		}
	}

	private static void printSequenceFile(Path path) {
		SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(
				path, configuration);
		for (Pair<Writable, Writable> pair : iterable) {
			System.out
					.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
		}
	}
}