package com.technobium;

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

		String outputFolder = "output/";
		Path documentsSequencePath = new Path(outputFolder, "sequence");

		// 1) create documents
		{
			SequenceFile.Writer writer = new SequenceFile.Writer(
					FileSystem.get(new Configuration()), new Configuration(),
					documentsSequencePath, Text.class, Text.class);

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

		// 2) calculate TF IDF
		{
			Path tokenizedDocumentsPath2 = new Path(outputFolder,
					DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
			DocumentProcessor.tokenizeDocuments(documentsSequencePath,
					StandardAnalyzer.class, tokenizedDocumentsPath2,
					new Configuration());

			DictionaryVectorizer.createTermFrequencyVectors(
					tokenizedDocumentsPath2, new Path(outputFolder),
					DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
					new Configuration(), 1, 1, 0.0f,
					PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false,
					false);

			Path termFrequencyVectorsPath2 = new Path(outputFolder
					+ DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER);
			Path tfidfPath2 = new Path(outputFolder + "tfidf");
			Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter
					.calculateDF(termFrequencyVectorsPath2, tfidfPath2,
							new Configuration(), 100);

			TFIDFConverter.processTfIdf(termFrequencyVectorsPath2, tfidfPath2,
					new Configuration(), documentFrequencies, 1, 100,
					PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
		}
		// 3) Cluster documents
		{
			String vectorsFolder2 = outputFolder + "tfidf/tfidf-vectors/";
			String canopyCentroids2 = outputFolder + "canopy-centroids";
			String clusterOutput2 = outputFolder + "clusters";
			Configuration configuration2 = new Configuration();
			if (FileSystem.get(configuration2).exists(new Path(clusterOutput2))) {
				FileSystem.get(configuration2).delete(new Path(clusterOutput2),
						true);
			}
			{
				// CosineDistanceMeasure
				CanopyDriver.run(new Path(vectorsFolder2), new Path(
						canopyCentroids2), new CosineDistanceMeasure(), 0.2,
						0.2, true, 1, true);

				FuzzyKMeansDriver.run(new Path(vectorsFolder2), new Path(
						canopyCentroids2, "clusters-0-final"), new Path(
						clusterOutput2), 0.01, 20, 2, true, true, 0, false);
			}
		}
		// 4) Print documents
		{
			for (Pair<Writable, Writable> pair : new SequenceFileIterable<Writable, Writable>(
					documentsSequencePath, new Configuration())) {
				System.out.format("%10s -> %s\n", pair.getFirst(),
						pair.getSecond());
			}
		}
		System.out.println("\n Clusters: ");

		// 5) Print clsuters
		{
			for (Pair<Writable, Writable> pair : new SequenceFileIterable<Writable, Writable>(
					new Path(outputFolder
							+ "clusters/clusteredPoints/part-m-00000"),
					new Configuration())) {
				System.out.format("%10s -> %s\n", pair.getFirst(),
						pair.getSecond());
			}
		}
	}
}