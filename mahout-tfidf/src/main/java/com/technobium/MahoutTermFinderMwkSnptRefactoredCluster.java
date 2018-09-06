package com.technobium;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
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
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;

/**
 * TFIDF (term frequency / document frequency) - for use on small *mwk files
 * 
 * Damn, this is failing too.
 */
// This was un-abstracted so that we can try and find phrases for the clustering
// code.
public class MahoutTermFinderMwkSnptRefactoredCluster {

	public static void main(final String[] args) throws Exception {
		// TODO: after finding the logic that is common to both, perform the
		// clustering
		// doClustering();
		// TOOD: I think I've done this wrong. I shouldn't be adding the
		// category
		// anywhere.
		// https://github.com/technobium/mahout-tfidf
		doTermFinding();
	}

	private static void doTermFinding() throws Exception {

		System.setProperty("org.apache.commons.logging.Log",
				"org.apache.commons.logging.impl.NoOpLog");

		Configuration configuration = new Configuration();
		String tempIntermediate = "temp_intermediate/";

		Path documentsSequencePath1 = writeToSequenceFile(
				configuration,
				new Path(tempIntermediate, "sequence"),
				new String[] {
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/aspergers",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/atletico",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/business",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/career",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/equalizer",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/productivity",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/self",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/self/approval_attention_social_status",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/self/cliquology_and_bullying/",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/soccer",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/tech/programming_tips",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/tech/programming_tips/functional_programming",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/travel",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/video_editing",
						System.getProperty("user.home")
								+ "/sarnobat.git/mwk/snippets/wrestling", });
		{

			Map<String, String> sequencesMap = sequenceFileToMap(configuration,
					documentsSequencePath1);

			// Too much output
			if (false) {
				for (String sequenceKey : sequencesMap.keySet()) {
					System.out
							.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - "
									+ documentsSequencePath1
									+ " - key="
									+ sequenceKey
									+ ", value="
									+ sequencesMap.get(sequenceKey).replaceAll(
											"\\n", ""));
				}
			}
		}

		{
			{

				Files.deleteIfExists(Paths
						.get("temp_intermediate/tokenized-documents/part-m-00000"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tf-vectors/part-r-00000"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/wordcount/part-r-00000"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/df-count/part-r-00000"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/frequency.file-0"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/dictionary.file-0"));

				Files.deleteIfExists(Paths
						.get("temp_intermediate/tokenized-documents/_SUCCESS"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tf-vectors/_SUCCESS"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/wordcount/_SUCCESS"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/df-count/_SUCCESS"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS"));
				Files.deleteIfExists(Paths
						.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS"));
			}
			// No files created so far.
			Path tokenizedDocumentsPath;
			try {
				tokenizedDocumentsPath = tokenizeDocuments(configuration,
						tempIntermediate, documentsSequencePath1);
			} catch (Exception e) {
				// IllegalStateException could get thrown I think, so we need
				// this
				e.printStackTrace();
				System.err
						.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Could not instantiate "
								+ MyEnglishAnalyzer.class
								+ ". Probably there is no public class and constructor.");
				return;
			}
			Preconditions.checkState(Paths
					.get("temp_intermediate/tokenized-documents/part-m-00000")
					.toFile().exists());
			Preconditions.checkState(Paths
					.get("temp_intermediate/tokenized-documents/_SUCCESS")
					.toFile().exists());

			{
				Path termFrequencies = new Path(
						"temp_intermediate/tokenized-documents/part-m-00000");
				Map<String, String> map = termFrequencyVectorsToMap(
						configuration, termFrequencies);
				for (String term : map.keySet()) {
					System.out
							.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() = "
									+ documentsSequencePath1
									+ " documents::terms : category="
									+ term
									+ "; terms=" + map.get(term));
				}
			}

			Preconditions.checkState(!Paths
					.get("temp_intermediate/dictionary.file-0").toFile()
					.exists());
			Path documentVectorOutputFolderPath = createTermFrequencyVectors(
					configuration, tempIntermediate, tokenizedDocumentsPath);
			Preconditions.checkState("temp_intermediate/tf-vectors"
					.equals(documentVectorOutputFolderPath.toString()),
					documentVectorOutputFolderPath);
			Preconditions.checkState(Paths
					.get("temp_intermediate/dictionary.file-0").toFile()
					.exists());
			{
				Path path = new Path("temp_intermediate/dictionary.file-0");
				Map<Integer, String> map = dictionaryToMap(configuration, path);
				// too much output
				if (false) {
					for (int term : map.keySet()) {
						System.out
								.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - dictionary : term_id="
										+ term + "; term=" + map.get(term));
					}
				}
			}
			{
				Path termFrequencies = new Path(
						"temp_intermediate/tf-vectors/part-r-00000");
				Map<String, String> map = termFrequencyVectorsToMap(
						configuration, termFrequencies);
				for (String term : map.keySet()) {
					System.out
							.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - tf vectors: category="
									+ term
									+ "; term frequencies="
									+ map.get(term));
				}
			}
			Preconditions.checkState(!Paths
					.get("temp_intermediate/tfidf/frequency.file-0").toFile()
					.exists());
			Preconditions.checkState(Paths
					.get("temp_intermediate/tf-vectors/_SUCCESS").toFile()
					.exists());
			Preconditions.checkState(Paths
					.get("temp_intermediate/tf-vectors/part-r-00000").toFile()
					.exists());
			Preconditions.checkState(Paths
					.get("temp_intermediate/wordcount/_SUCCESS").toFile()
					.exists());
			Preconditions.checkState(Paths
					.get("temp_intermediate/wordcount/part-r-00000").toFile()
					.exists());

			Preconditions.checkState(!Paths
					.get("temp_intermediate/tfidf/df-count/part-r-00000")
					.toFile().exists());
			Preconditions.checkState(!Paths
					.get("temp_intermediate/tfidf/df-count/_SUCCESS").toFile()
					.exists());
			Preconditions
					.checkState(!Paths
							.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000")
							.toFile().exists());
			Preconditions.checkState(!Paths
					.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS")
					.toFile().exists());
			Preconditions.checkState(!Paths
					.get("temp_intermediate/tfidf/frequency.file-0").toFile()
					.exists());
			// System.err.println("MahoutTermFinder.calculateTfIdf() - Creating
			// term vectors
			// using input file " + new Path(outputFolder +
			// DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));
			Path tfidfPath = new Path(tempIntermediate + "/tfidf/");
			System.err
					.println("MahoutTermFinder.calculateTfIdf() - adding document frequencies to file "
							+ tfidfPath);
			{
				System.err
						.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - "
								+ documentVectorOutputFolderPath + " ===> "
								+ tfidfPath);
				Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter
						.calculateDF(documentVectorOutputFolderPath, tfidfPath,
								configuration, 100);
				Preconditions.checkState(Paths
						.get("temp_intermediate/tfidf/df-count/part-r-00000")
						.toFile().exists());
				Preconditions.checkState(Paths
						.get("temp_intermediate/tfidf/df-count/_SUCCESS")
						.toFile().exists());
				Preconditions.checkState(Paths
						.get("temp_intermediate/tfidf/frequency.file-0")
						.toFile().exists());

				System.err
						.println("MahoutTermFinder.calculateTfIdf() - adding tfidf scores to file "
								+ tfidfPath);
				System.err
						.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - "
								+ documentVectorOutputFolderPath + " ===> "
								+ tfidfPath);
				Preconditions.checkState(!Paths
						.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS")
						.toFile().exists());
				Preconditions
						.checkState(!Paths
								.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000")
								.toFile().exists());
				Preconditions.checkState("temp_intermediate/tf-vectors"
						.equals(documentVectorOutputFolderPath.toString()),
						documentVectorOutputFolderPath);
				Preconditions.checkState(
						"temp_intermediate/tfidf".equals(tfidfPath.toString()),
						tfidfPath);
				TFIDFConverter.processTfIdf(documentVectorOutputFolderPath,
						tfidfPath, configuration, documentFrequencies, 1, 100,
						PartialVectorMerger.NO_NORMALIZING, false, false,
						false, 1);
				Preconditions.checkState(Paths
						.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS")
						.toFile().exists());
				Preconditions
						.checkState(Paths
								.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000")
								.toFile().exists());
				Preconditions
						.checkState(Paths
								.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000")
								.toFile().exists());
				Preconditions
						.checkState(!Paths
								.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS")
								.toFile().exists());
				Preconditions
						.checkState(!Paths
								.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000")
								.toFile().exists());
			}
		}
		Path dictionaryFilePath = new Path(tempIntermediate,
				"dictionary.file-0");
		Preconditions.checkState(Paths
				.get("temp_intermediate/dictionary.file-0").toFile().exists());

		System.err.println("MahoutTermFinder.main() - ??? ===> "
				+ dictionaryFilePath);
		System.err
				.println("MahoutTermFinder.main() - Reading dictionary into map. Dictionary of terms with IDs: "
						+ dictionaryFilePath + " (large)");
		{
			// Create a vector numerical value for each term (e.g. "atletico" ->
			// 4119)
			SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
					dictionaryFilePath, configuration);
			Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
			for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
				// System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - sequenceFile = "
				// + sequenceFile);
				// System.err.format("%10s -> %s\n", pair.getFirst(),
				// pair.getSecond());
				termToOrdinalMappings2.put(sequenceFile.getFirst().toString(),
						sequenceFile.getSecond());
			}
		}

		System.err.println("MahoutTermFinder.main() - Creating TFIDF Vectors");
		{
			// Create a vector numerical value for each term (e.g. "atletico" ->
			// 4119)
			Path tfIdfVectorsPath = new Path(tempIntermediate,
					"tfidf/tfidf-vectors/part-r-00000");
			SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
					tfIdfVectorsPath, configuration);
			Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
			for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
				// System.err.format("%10s -> %s\n", pair.getFirst(),
				// pair.getSecond());
				termToOrdinalMappings2.put(sequenceFile.getFirst().toString(),
						sequenceFile.getSecond());
			}
		}
		// 4) Print documents
		{
			org.apache.hadoop.fs.Path documentsSequencePath = new org.apache.hadoop.fs.Path(
					"temp_intermediate/tokenized-documents/part-m-00000");
			for (Pair<Writable, Writable> pair : new SequenceFileIterable<Writable, Writable>(
					documentsSequencePath, new Configuration())) {
				System.out.format("%10s -> %s\n", pair.getFirst(),
						pair.getSecond());
			}
		}

		// 5) Do clustering
		{
			System.out
					.println("MahoutTermClusterMwkSnpt.doTermFinding() - now run clustering");
			clusterDocuments(tempIntermediate);
		}

		System.err
				.println("MahoutTermFinderMwkSnptRefactored.doTermFinding() - hereafter, we deal exclusively with maps, not sequence files.");
	}

	private static void clusterDocuments(String tempIntermediate)
			throws IOException, InterruptedException, ClassNotFoundException {
		String outputFolder = tempIntermediate;
		// 3) Cluster documents
		{
			String vectorsFolder2 = outputFolder + "/tfidf/tfidf-vectors/";
			// check the distance between 2 documents at random, so we know the
			// values of t1 and t2 to use.
			DistanceMeasure distanceMeasure = new CosineDistanceMeasure();
			// DistanceMeasure distanceMeasure = new TanimotoDistanceMeasure();
			{
				SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
						new Path(vectorsFolder2 + "part-r-00000"),
						new Configuration());
				Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
				int i = 0;
				for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
					termToOrdinalMappings2.put(sequenceFile.getFirst()
							.toString(), sequenceFile.getSecond());
					if (i > 1) {
						break;
					}
				}

				Iterator<Object> iterator = termToOrdinalMappings2.values()
						.iterator();
				Vector v1 = ((VectorWritable) iterator.next()).get();
				Vector v2 = ((VectorWritable) iterator.next()).get();
				double distance = distanceMeasure.distance(v1, v2);
				System.out
						.println("MahoutTermFinderMwkSnptRefactoredCluster.clusterDocuments() v1 = "
								+ v1);
				System.out
						.println("MahoutTermFinderMwkSnptRefactoredCluster.clusterDocuments() v2 = "
								+ v2);
				System.out
						.println("MahoutTermFinderMwkSnptRefactoredCluster.clusterDocuments() distance = "
								+ distance);
			}
			String canopyCentroids2 = outputFolder + "/canopy-centroids";
			String clusterOutput2 = outputFolder + "/clusters";
			Configuration configuration2 = new Configuration();
			if (FileSystem.get(configuration2).exists(new Path(clusterOutput2))) {
				FileSystem.get(configuration2).delete(new Path(clusterOutput2),
						true);
			}
			{
				// CosineDistanceMeasure
				CanopyDriver.run(new Path(vectorsFolder2), new Path(
						canopyCentroids2), distanceMeasure, 0.9, 0.9, true, 1,
						true);

				FuzzyKMeansDriver.run(new Path(vectorsFolder2), new Path(
						canopyCentroids2, "clusters-0-final"), new Path(
						clusterOutput2), 0.01, 20, 2, true, true, 0, false);
			}
		}
		System.out.println("\n Clusters: ");

		// 5) Print clusters
		{
			Multimap<String, String> clusterToDocuments = HashMultimap.create();
			for (Pair<Writable, Writable> pair : new SequenceFileIterable<Writable, Writable>(
					new Path(outputFolder
							+ "clusters/clusteredPoints/part-m-00000"),
					new Configuration())) {
				Writable first = pair.getFirst();
				System.out.format("%10s -> %s\n", first, pair.getSecond()
						.getClass());
				// System.out
				// .println("MahoutTermFinderMwkSnptRefactoredCluster.clusterDocuments() first = "
				// + first);
				String documentID = ((NamedVector) ((WeightedPropertyVectorWritable) pair
						.getSecond()).getVector()).getName();
				// System.out
				// .println("MahoutTermFinderMwkSnptRefactoredCluster.clusterDocuments() second = "
				// + documentID);
				if (first instanceof WeightedPropertyVectorWritable) {
					clusterToDocuments.put((first).toString(), documentID);
				} else if (first instanceof IntWritable) {
					clusterToDocuments.put(((IntWritable) first).toString(),
							documentID);
				}
			}

			for (String clusterID : clusterToDocuments.keySet()) {
				System.out
						.println("MahoutTermFinderMwkSnptRefactoredCluster.clusterDocuments() "
								+ clusterID
								+ " :: "
								+ clusterToDocuments.get(clusterID));
			}
		}
	}

	private static Map<String, String> sequenceFileToMap(
			Configuration configuration, Path documentsSequencePath1)
			throws IOException {
		Map<String, String> sequencesMap = new HashMap<String, String>();
		Path path = documentsSequencePath1;
		Configuration conf = configuration;
		FileSystem fs = FileSystem.getLocal(conf);
		SequenceFile.Reader reader = null;
		try {
			reader = new SequenceFile.Reader(fs, path, conf);
			Writable key = (Writable) ReflectionUtils.newInstance(
					reader.getKeyClass(), conf);
			Writable value = (Writable) ReflectionUtils.newInstance(
					reader.getValueClass(), conf);
			while (reader.next(key, value)) {
				// System.out.println("Key: " + key + " value:" + value);
				sequencesMap.put(key.toString(), value.toString());
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		} finally {
			reader.close();
		}
		return sequencesMap;
	}

	private static Map<Integer, String> dictionaryToMap(
			Configuration configuration, Path path) throws IOException {
		FileSystem fs = FileSystem.getLocal(configuration);
		SequenceFile.Reader read = new SequenceFile.Reader(fs, path,
				configuration);
		// To sort the entries
		Map<Integer, String> map = new TreeMap<Integer, String>();
		Text text = new Text();
		IntWritable dicKey = new IntWritable();
		while (read.next(text, dicKey)) {
			map.put(Integer.parseInt(dicKey.toString()), text.toString());
		}
		read.close();
		return map;
	}

	private static Map<String, String> termFrequencyVectorsToMap(
			Configuration configuration, Path path) throws IOException {
		Map<String, String> termFrequenciesMap = new HashMap<String, String>();
		SequenceFile.Reader reader = null;
		try {

			FileSystem fs = FileSystem.getLocal(configuration);
			reader = new SequenceFile.Reader(fs, path, configuration);
			Writable key = (Writable) ReflectionUtils.newInstance(
					reader.getKeyClass(), configuration);
			Writable value = (Writable) ReflectionUtils.newInstance(
					reader.getValueClass(), configuration);
			// long position = reader.getPosition();
			while (reader.next(key, value)) {
				termFrequenciesMap.put(key.toString(), value.toString());
				// System.out.println("TF vector: Key: " + key + " value:" +
				// value);
				// position = reader.getPosition();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		} finally {
			reader.close();
		}
		return termFrequenciesMap;
	}

	private static Path createTermFrequencyVectors(Configuration configuration,
			String outputFolder, Path tokenizedDocumentsPath)
			throws IOException, InterruptedException, ClassNotFoundException {
		String documentVectorOutputFolder = createTermFrequencyVectors1(
				configuration, outputFolder, tokenizedDocumentsPath);
		Path documentVectorOutputFolderPath = new Path(outputFolder
				+ documentVectorOutputFolder);
		return documentVectorOutputFolderPath;
	}

	// TODO: inline this
	private static String createTermFrequencyVectors1(
			Configuration configuration, String outputFolder,
			Path tokenizedDocumentsPath) throws IOException,
			InterruptedException, ClassNotFoundException {
		String documentVectorOutputFolder = DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER;
		System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - "
				+ tokenizedDocumentsPath + " ===> "
				+ new Path(outputFolder + "/" + documentVectorOutputFolder));
		DictionaryVectorizer.createTermFrequencyVectors(
				tokenizedDocumentsPath,
				new Path(outputFolder),
				// TODO: use documentVectorOutputFolder, not
				// DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
				configuration, 1, 1, 0.0f, PartialVectorMerger.NO_NORMALIZING,
				true, 1, 100, false, false);
		return documentVectorOutputFolder;
	}

	private static Path tokenizeDocuments(Configuration configuration,
			String outputFolder, Path documentsSequencePath)
			throws IOException, InterruptedException, ClassNotFoundException {
		Path tokenizedDocumentsPath = new Path(outputFolder,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
		System.err
				.println("SRIDHAR MahoutTermFinderMwkSnpt.tokenizeDocuments() - Adding tokenized documents to folder "
						+ tokenizedDocumentsPath);
		System.err
				.println("MahoutTermFinderMwkSnpt.tokenizeDocuments() - Tokenzing documents, using "
						+ MyEnglishAnalyzer.class
						+ " using reflection (yuck). Outputting to: "
						+ tokenizedDocumentsPath);
		System.err
				.println("SRIDHAR MahoutTermFinderMwkSnpt.tokenizeDocuments() - "
						+ documentsSequencePath
						+ " ===> "
						+ tokenizedDocumentsPath);
		DocumentProcessor.tokenizeDocuments(documentsSequencePath,
				MyEnglishAnalyzer.class, tokenizedDocumentsPath, configuration);
		return tokenizedDocumentsPath;
	}

	private static Path writeToSequenceFile(Configuration configuration,
			Path documentsSequencePath, String[] mwkSnippetCategoryDirs)
			throws IOException {
		System.err
				.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Creating sequence file from mwk snippet files, outputting files to sequence file "
						+ documentsSequencePath + " (large)");
		SequenceFile.Writer writer = new SequenceFile.Writer(
				FileSystem.get(configuration), configuration,
				documentsSequencePath, Text.class, Text.class);

		for (String path : mwkSnippetCategoryDirs) {
			DirectoryStream<java.nio.file.Path> stream = Files
					.newDirectoryStream(Paths.get(path));
			try {
				int i = 0;
				int total = 0;
				for (java.nio.file.Path fileInPath : stream) {
					if (Files.isDirectory(fileInPath)) {
						// listFiles(entry);
						System.err
								.println("SRIDHAR MahoutTermClusterMwkSnpt.writeToSequenceFile() - skipping nested dir: "
										+ fileInPath);
					} else {
						if (fileInPath.toFile().exists()) {
							Text cateogoryDir = new Text(fileInPath
									.getFileName().toString());
							String readFileToString = FileUtils
									.readFileToString(Paths.get(
											fileInPath.toUri()).toFile());
							if (i % 100 == 0) {
								System.err
										.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - "
												+ cateogoryDir
												+ "::"
												+ StringUtils
														.substring(
																readFileToString,
																0, 30));
							}
							// System.out.println("SRIDHAR
							// MahoutTermClusterMwkSnpt.writeToSequenceFile() -
							// writing to sequence file: " + fileInPath);
							if (total > 2) {
								break;
							}
							writer.append(cateogoryDir, new Text(
									readFileToString));
							total++;
							// TODO: this is wrong, it's overwriting previous
							// files in the same dir
							// I shouldn't be adding the category anywhere.
							// https://github.com/technobium/mahout-tfidf
						}
					}
					++i;
				}
			} catch (IOException e3) {
				throw e3;
			} finally {
			}
		}

		writer.close();
		return documentsSequencePath;
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

		public MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords,
				CharArraySet stemExclusionSet) {
			super(matchVersion, stopwords);
			this.stemExclusionSet = CharArraySet.unmodifiableSet(CharArraySet
					.copy(matchVersion, stemExclusionSet));
		}

		@Override
		protected TokenStreamComponents createComponents(String fieldName,
				Reader reader) {
			final Tokenizer source = new StandardTokenizer(matchVersion, reader);
			TokenStream result = new StandardFilter(matchVersion, source);
			// prior to this we get the classic behavior, standardfilter does it
			// for
			// us.
			if (matchVersion.onOrAfter(Version.LUCENE_31))
				result = new EnglishPossessiveFilter(matchVersion, result);
			result = new LowerCaseFilter(matchVersion, result);
			CharArraySet stopwords2;
			try {
				stopwords2 = getStopWords(System.getProperty("user.home")
						+ "/github/mahout/stopwords.txt");
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

		private static CharArraySet getStopWords(String stoplist)
				throws IOException {
			List<String> ss = FileUtils.readLines(Paths.get(stoplist).toFile());
			CharArraySet ret = new CharArraySet(Version.LUCENE_CURRENT, ss,
					false);
			ret.addAll(ss);
			return ret;
		}
	}
}
