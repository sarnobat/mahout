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


	public static void main(String args[]) throws Exception {


		String outputFolder;
		Configuration configuration;
		FileSystem fileSystem;
		Path documentsSequencePath;
		Path tokenizedDocumentsPath;
		Path tfidfPath;
		Path termFrequencyVectorsPath;
		
		configuration = new Configuration();
		fileSystem = FileSystem.get(configuration);

		outputFolder = "output/";
		documentsSequencePath = new Path(outputFolder, "sequence");
		tokenizedDocumentsPath = new Path(outputFolder,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
		tfidfPath = new Path(outputFolder + "tfidf");
		termFrequencyVectorsPath = new Path(outputFolder
				+ DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER);

		createTestDocuments(fileSystem, documentsSequencePath, configuration);
		calculateTfIdf(documentsSequencePath, tokenizedDocumentsPath, configuration, termFrequencyVectorsPath, tfidfPath, outputFolder);
		clusterDocs(outputFolder + "tfidf/tfidf-vectors/", outputFolder + "canopy-centroids", outputFolder + "clusters", configuration);

		printSequenceFile(documentsSequencePath, configuration);

		System.out.println("\n Clusters: ");
		printSequenceFile(new Path(outputFolder
				+ "clusters/clusteredPoints/part-m-00000"), configuration);
	}

	private static void createTestDocuments(FileSystem fileSystem3, Path documentsSequencePath, Configuration configuration3) throws IOException {
		FileSystem fileSystem2 = fileSystem3;
		Path documentsSequencePath2 = documentsSequencePath;
		Configuration configuration2 = configuration3;
		SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem2,
				configuration2, documentsSequencePath2, Text.class, Text.class);

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

	private static void calculateTfIdf(Path documentsSequencePath3, Path tokenizedDocumentsPath3, Configuration configuration3, Path termFrequencyVectorsPath3, Path tfidfPath3, String outputFolder3) throws ClassNotFoundException,
			IOException, InterruptedException {
		Path documentsSequencePath2 = documentsSequencePath3;
		Path tokenizedDocumentsPath2 = tokenizedDocumentsPath3;
		Configuration configuration2 = configuration3;
		DocumentProcessor.tokenizeDocuments(documentsSequencePath2,
				StandardAnalyzer.class, tokenizedDocumentsPath2, configuration2);

		String outputFolder2 = outputFolder3;
		DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath2,
				new Path(outputFolder2),
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
				configuration2, 1, 1, 0.0f, PartialVectorMerger.NO_NORMALIZING,
				true, 1, 100, false, false);

		Path termFrequencyVectorsPath2 = termFrequencyVectorsPath3;
		Path tfidfPath2 = tfidfPath3;
		Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter
				.calculateDF(termFrequencyVectorsPath2, tfidfPath2,
						configuration2, 100);

		TFIDFConverter.processTfIdf(termFrequencyVectorsPath2, tfidfPath2,
				configuration2, documentFrequencies, 1, 100,
				PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
	}

	private static void clusterDocs(String vectorsFolder2, String canopyCentroids2, String clusterOutput2, Configuration configuration2) throws ClassNotFoundException,
			IOException, InterruptedException {
		String vectorsFolder = vectorsFolder2;
		String canopyCentroids = canopyCentroids2;
		String clusterOutput = clusterOutput2;

		FileSystem fs = FileSystem.get(configuration2);
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

	private static void printSequenceFile(Path path, Configuration configuration3) {
		Configuration configuration2 = configuration3;
		SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(
				path, configuration2);
		for (Pair<Writable, Writable> pair : iterable) {
			System.out
					.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
		}
	}
}