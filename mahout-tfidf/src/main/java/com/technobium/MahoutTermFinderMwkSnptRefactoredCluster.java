package com.technobium;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
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
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * TFIDF (term frequency / document frequency) - for use on small *mwk files
 * 
 * Damn, this is failing too.
 */
// This was un-abstracted so that we can try and find phrases for the clustering
// code.
public class MahoutTermFinderMwkSnptRefactoredCluster {

    private static final Logger LOG = LoggerFactory.getLogger(MahoutTermFinderMwkSnptRefactoredCluster.class);

    public static void main(final String[] args) throws Exception {
        // TODO: after finding the logic that is common to both, perform the clustering
        // doClustering();
        // TOOD: I think I've done this wrong. I shouldn't be adding the category
        // anywhere.
        // https://github.com/technobium/mahout-tfidf
        doTermFinding();
    }

    private static void writePointsToFile(final Configuration configuration, final List<MwkVector> points,
            Path pointsFile) throws IOException {
        System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - begin");
        FileSystem fs = FileSystem.getLocal(configuration);
        System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 1");
        SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, pointsFile, IntWritable.class,
                VectorWritable.class);
        System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 2");

        int recNum = 0;
        final VectorWritable vec = new VectorWritable();

        System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 3");
        for (final MwkVector point : points) {
            System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - point = " + point);
            vec.set(point.getVector());
            writer.append(new IntWritable(recNum++), vec);
        }
        System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - end");
        writer.close();
    }

    private static void writeClusterInitialCenters(final Configuration configuration, final List<MwkVector> points,
            String clusterPath, int clusterDesiredCount, Path clusterOutputFilePath) throws IOException {
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writeClusterInitialCenters() - ");
        final Path writerPath = clusterOutputFilePath;

        FileSystem fs = FileSystem.getLocal(configuration);
        // final Path path = new Path(POINTS_PATH + "/pointsFile");
        final SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, writerPath, Text.class,
                Kluster.class);

        for (int i = 0; i < clusterDesiredCount; i++) {
            final MwkVector vec = points.get(i);

            // write the initial centers
            final Kluster cluster = new Kluster(vec.getVector(), i, new EuclideanDistanceMeasure());
            writer.append(new Text(cluster.getIdentifier()), cluster);
            System.out.println(
                    "SRIDHAR MahoutTermClusterMwkSnpt.writeClusterInitialCenters() - cluster = " + cluster.toString());
        }

        writer.close();
    }

    private static void readAndPrintOutputValues(final Configuration configuration, Path clusteredPointsInputPath)
            throws IOException {
        FileSystem fs = FileSystem.getLocal(configuration);
        final SequenceFile.Reader reader = new SequenceFile.Reader(fs, clusteredPointsInputPath, configuration);

        final IntWritable key = new IntWritable();
        final WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
        int count = 0;
        while (reader.next(key, value)) {
            System.out.printf(
                    "SRIDHAR MahoutTermClusterMwkSnpt.readAndPrintOutputValues() - " + "%s belongs to cluster %s\n",
                    value.toString(), key.toString());
            LOG.info("{} belongs to cluster {}", value.toString(), key.toString());
            count++;
        }
        reader.close();
        if (count == 0) {
            throw new RuntimeException("No  output pairs");
        }
    }

    // Read the points to vector from 2D array
    private static List<MwkVector> vectorize(final double[][] raw) {
        System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.vectorize() - ");
        final List<MwkVector> points = new ArrayList<MwkVector>();

        for (int i = 0; i < raw.length; i++) {
            MwkVector vec = new MwkVector(new RandomAccessSparseVector(raw[i].length));
            vec.getVector().assign(raw[i]);
            points.add(vec);
        }

        return points;
    }

    private static class MwkVector {

        private final Vector vector;

        @Deprecated
        MwkVector(Vector vector) {
            this.vector = vector;
        }

        MwkVector(Path mwkFile) {
            this.vector = toVector(mwkFile);
        }

        private static Vector toVector(Path fileInPath) {
            // TODO Auto-generated method stub
            // return null;
            File file = Paths.get(fileInPath.toUri()).toFile();
            if (file.exists()) {
                try {
                    Text text = new Text(FileUtils.readFileToString(file));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                // writer.append(id, text);
            }
            throw new RuntimeException("Finish implementing this");
        }

        // TODO: this violates Demeter. Fix later once we have it working.
        Vector getVector() {
            return vector;
        }
    }

    private static final int THRESHOLD = 1;

    private static void doTermFinding() throws Exception {

        System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog");

        Configuration configuration = new Configuration();
        String tempIntermediate = "temp_intermediate/";

        Path documentsSequencePath1 = writeToSequenceFile(configuration, new Path(tempIntermediate, "sequence"),
                new String[] { System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/aspergers",
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
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/wrestling", });
        {

            Map<String, String> sequencesMap = sequenceFileToMap(configuration, documentsSequencePath1);

            for (String sequenceKey : sequencesMap.keySet()) {
                System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - " + documentsSequencePath1
                        + " - key=" + sequenceKey + ", value=" + sequencesMap.get(sequenceKey).replaceAll("\\n", ""));
            }
            System.out.println(
                    "SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() -  TOOD: I've done this wrong. I shouldn't be adding the category anywhere. https://github.com/technobium/mahout-tfidf . Skip the TFIDF example and go straight to this better clustering example than the existing one I based on: https://github.com/technobium/mahout-clustering");
        }
        
        {
            {

                Files.deleteIfExists(Paths.get("temp_intermediate/tokenized-documents/part-m-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tf-vectors/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/wordcount/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/df-count/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/frequency.file-0"));
                Files.deleteIfExists(Paths.get("temp_intermediate/dictionary.file-0"));

                Files.deleteIfExists(Paths.get("temp_intermediate/tokenized-documents/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tf-vectors/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/wordcount/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/df-count/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS"));
            }
            // No files created so far.
            Path tokenizedDocumentsPath;
            try {
                tokenizedDocumentsPath = tokenizeDocuments(configuration, tempIntermediate, documentsSequencePath1);
            } catch (Exception e) {
                // IllegalStateException could get thrown I think, so we need this
                e.printStackTrace();
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Could not instantiate "
                        + MyEnglishAnalyzer.class + ". Probably there is no public class and constructor.");
                return;
            }
            Preconditions.checkState(Paths.get("temp_intermediate/tokenized-documents/part-m-00000").toFile().exists());
            Preconditions.checkState(Paths.get("temp_intermediate/tokenized-documents/_SUCCESS").toFile().exists());

            {
                Path termFrequencies = new Path("temp_intermediate/tokenized-documents/part-m-00000");
                Map<String, String> map = termFrequencyVectorsToMap(configuration, termFrequencies);
                for (String term : map.keySet()) {
                    System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() = " + documentsSequencePath1
                            + " documents::terms : category=" + term + "; terms=" + map.get(term));
                }
            }

            Preconditions.checkState(!Paths.get("temp_intermediate/dictionary.file-0").toFile().exists());
            Path documentVectorOutputFolderPath = createTermFrequencyVectors(configuration, tempIntermediate,
                    tokenizedDocumentsPath);
            Preconditions.checkState("temp_intermediate/tf-vectors".equals(documentVectorOutputFolderPath.toString()), documentVectorOutputFolderPath);
            Preconditions.checkState(Paths.get("temp_intermediate/dictionary.file-0").toFile().exists());
            {
                Path path = new Path("temp_intermediate/dictionary.file-0");
                Map<Integer, String> map = dictionaryToMap(configuration, path);
                for (int term : map.keySet()) {
                    System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - dictionary : term_id=" + term
                            + "; term=" + map.get(term));
                }
            }
            {
                Path termFrequencies = new Path(
                        "temp_intermediate/tf-vectors/part-r-00000");
                Map<String, String> map = termFrequencyVectorsToMap(configuration, termFrequencies);
                for (String term : map.keySet()) {
                    System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - tf vectors: category=" + term
                            + "; term frequencies=" + map.get(term));
                }
            }
            Preconditions.checkState(!Paths.get("temp_intermediate/tfidf/frequency.file-0").toFile().exists());
            Preconditions.checkState(Paths.get("temp_intermediate/tf-vectors/_SUCCESS").toFile().exists());
            Preconditions.checkState(Paths.get("temp_intermediate/tf-vectors/part-r-00000").toFile().exists());
            Preconditions.checkState(Paths.get("temp_intermediate/wordcount/_SUCCESS").toFile().exists());
            Preconditions.checkState(Paths.get("temp_intermediate/wordcount/part-r-00000").toFile().exists());

            Preconditions.checkState(!Paths.get("temp_intermediate/tfidf/df-count/part-r-00000").toFile().exists());
            Preconditions.checkState(!Paths.get("temp_intermediate/tfidf/df-count/_SUCCESS").toFile().exists());
            Preconditions
                    .checkState(!Paths.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000").toFile().exists());
            Preconditions
                    .checkState(!Paths.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS").toFile().exists());
            Preconditions.checkState(!Paths.get("temp_intermediate/tfidf/frequency.file-0").toFile().exists());
            // System.err.println("MahoutTermFinder.calculateTfIdf() - Creating term vectors
            // using input file " + new Path(outputFolder +
            // DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER));
            Path tfidfPath = new Path(tempIntermediate + "/tfidf/");
            System.err.println("MahoutTermFinder.calculateTfIdf() - adding document frequencies to file " + tfidfPath);
            {
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + documentVectorOutputFolderPath
                        + " ===> " + tfidfPath);
                Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter
                        .calculateDF(documentVectorOutputFolderPath, tfidfPath, configuration, 100);
                Preconditions.checkState(Paths.get("temp_intermediate/tfidf/df-count/part-r-00000").toFile().exists());
                Preconditions.checkState(Paths.get("temp_intermediate/tfidf/df-count/_SUCCESS").toFile().exists());
                Preconditions.checkState(Paths.get("temp_intermediate/tfidf/frequency.file-0").toFile().exists());

                System.err.println("MahoutTermFinder.calculateTfIdf() - adding tfidf scores to file " + tfidfPath);
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + documentVectorOutputFolderPath
                        + " ===> " + tfidfPath);
                Preconditions
                        .checkState(!Paths.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS").toFile().exists());
                Preconditions
                        .checkState(!Paths.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000").toFile().exists());
                Preconditions.checkState("temp_intermediate/tf-vectors".equals(documentVectorOutputFolderPath.toString()), documentVectorOutputFolderPath);
                Preconditions.checkState("temp_intermediate/tfidf".equals(tfidfPath.toString()), tfidfPath);
                TFIDFConverter.processTfIdf(documentVectorOutputFolderPath, tfidfPath, configuration,
                        documentFrequencies, 1, 100, PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);
                Preconditions.checkState(Paths.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS").toFile().exists());
                Preconditions.checkState(Paths.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000").toFile().exists());
                Preconditions
                        .checkState(Paths.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000").toFile().exists());
                Preconditions
                        .checkState(!Paths.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS").toFile().exists());
                Preconditions.checkState(
                        !Paths.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000").toFile().exists());
            }
        }
        Path dictionaryFilePath = new Path(tempIntermediate, "dictionary.file-0");
        Preconditions.checkState(Paths.get("temp_intermediate/dictionary.file-0").toFile().exists());

        System.err.println("MahoutTermFinder.main() - ??? ===> " + dictionaryFilePath);
        System.err.println("MahoutTermFinder.main() - Reading dictionary into map. Dictionary of terms with IDs: "
                + dictionaryFilePath + " (large)");
        Map<String, Object> dictionary;
        {
            // Create a vector numerical value for each term (e.g. "atletico" -> 4119)
            SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
                    dictionaryFilePath, configuration);
            Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
            for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
                // System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - sequenceFile = "
                // + sequenceFile);
                // System.err.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
                termToOrdinalMappings2.put(sequenceFile.getFirst().toString(), sequenceFile.getSecond());
            }
            dictionary = termToOrdinalMappings2;
        }

        System.err.println("MahoutTermFinder.main() - Creating TFIDF Vectors");
        Map<String, Object> tfidf;
        {
            // Create a vector numerical value for each term (e.g. "atletico" -> 4119)
            Path tfIdfVectorsPath = new Path(tempIntermediate, "tfidf/tfidf-vectors/part-r-00000");
            SequenceFileIterable<Writable, Writable> sequenceFiles2 = new SequenceFileIterable<Writable, Writable>(
                    tfIdfVectorsPath, configuration);
            Map<String, Object> termToOrdinalMappings2 = new HashMap<String, Object>();
            for (Pair<Writable, Writable> sequenceFile : sequenceFiles2) {
                // System.err.format("%10s -> %s\n", pair.getFirst(), pair.getSecond());
                termToOrdinalMappings2.put(sequenceFile.getFirst().toString(), sequenceFile.getSecond());
            }
            tfidf = termToOrdinalMappings2;
        }

		{
			System.out
					.println("MahoutTermClusterMwkSnpt.doTermFinding() - now run clustering");
			clusterDocuments(tempIntermediate);
		}
		
        System.err.println("MahoutTermFinderMwkSnptRefactored.doTermFinding() - hereafter, we deal exclusively with maps, not sequence files.");
    }


	private static void clusterDocuments(String tempIntermediate)
			throws IOException, InterruptedException, ClassNotFoundException {
		String outputFolder = tempIntermediate;
		// 3) Cluster documents
		{
			String vectorsFolder2 = outputFolder + "/tfidf/tfidf-vectors/";
			String canopyCentroids2 = outputFolder + "/canopy-centroids";
			String clusterOutput2 = outputFolder + "/clusters";
			Configuration configuration2 = new Configuration();
			if (FileSystem.get(configuration2).exists(
					new Path(clusterOutput2))) {
				FileSystem.get(configuration2).delete(
						new Path(clusterOutput2), true);
			}
			{
				// CosineDistanceMeasure
				CanopyDriver.run(new Path(vectorsFolder2), new Path(
						canopyCentroids2), new CosineDistanceMeasure(),
						0.2, 0.2, true, 1, true);

				FuzzyKMeansDriver.run(new Path(vectorsFolder2), new Path(
						canopyCentroids2, "clusters-0-final"), new Path(
						clusterOutput2), 0.01, 20, 2, true, true, 0, false);
			}
		}
		// 4) Print documents
		{
			org.apache.hadoop.fs.Path documentsSequencePath = new org.apache.hadoop.fs.Path(
					"temp_intermediate/tokenized-documents/");
			for (Pair<Writable, Writable> pair : new SequenceFileIterable<Writable, Writable>(
					documentsSequencePath, new Configuration())) {
				System.out.format("%10s -> %s\n", pair.getFirst(),
						pair.getSecond());
			}
		}
		System.out.println("\n Clusters: ");

		// 5) Print clusters
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

	private static Map<String, String> sequenceFileToMap(Configuration configuration, Path documentsSequencePath1)
            throws IOException {
        Map<String, String> sequencesMap = new HashMap<String, String>();
        Path path = documentsSequencePath1;
        Configuration conf = configuration;
        FileSystem fs = FileSystem.getLocal(conf);
        SequenceFile.Reader reader = null;
        try {
            reader = new SequenceFile.Reader(fs, path, conf);
            Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
            Writable value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), conf);
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

    private static Map<Integer, String> dictionaryToMap(Configuration configuration, Path path) throws IOException {
        FileSystem fs = FileSystem.getLocal(configuration);
        SequenceFile.Reader read = new SequenceFile.Reader(fs, path, configuration);
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

    private static Map<String, String> termFrequencyVectorsToMap(Configuration configuration, Path path)
            throws IOException {
        Map<String, String> termFrequenciesMap = new HashMap<String, String>();
        SequenceFile.Reader reader = null;
        try {

            FileSystem fs = FileSystem.getLocal(configuration);
            reader = new SequenceFile.Reader(fs, path, configuration);
            Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), configuration);
            Writable value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), configuration);
            long position = reader.getPosition();
            while (reader.next(key, value)) {
                termFrequenciesMap.put(key.toString(), value.toString());
                // System.out.println("TF vector: Key: " + key + " value:" + value);
                position = reader.getPosition();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        } finally {
            reader.close();
        }
        return termFrequenciesMap;
    }

    private static Path createTermFrequencyVectors(Configuration configuration, String outputFolder,
            Path tokenizedDocumentsPath) throws IOException, InterruptedException, ClassNotFoundException {
        String documentVectorOutputFolder = createTermFrequencyVectors1(configuration, outputFolder,
                tokenizedDocumentsPath);
        Path documentVectorOutputFolderPath = new Path(outputFolder + documentVectorOutputFolder);
        return documentVectorOutputFolderPath;
    }

    // TODO: inline this
    private static String createTermFrequencyVectors1(Configuration configuration, String outputFolder,
            Path tokenizedDocumentsPath) throws IOException, InterruptedException, ClassNotFoundException {
        String documentVectorOutputFolder = DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER;
        System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - " + tokenizedDocumentsPath + " ===> "
                + new Path(outputFolder + "/" + documentVectorOutputFolder));
        DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, new Path(outputFolder),
                // TODO: use documentVectorOutputFolder, not
                // DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER
                DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, configuration, 1, 1, 0.0f,
                PartialVectorMerger.NO_NORMALIZING, true, 1, 100, false, false);
        return documentVectorOutputFolder;
    }

    private static Path tokenizeDocuments(Configuration configuration, String outputFolder, Path documentsSequencePath)
            throws IOException, InterruptedException, ClassNotFoundException {
        Path tokenizedDocumentsPath = new Path(outputFolder, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
        System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.tokenizeDocuments() - Adding tokenized documents to folder "
                + tokenizedDocumentsPath);
        System.err.println("MahoutTermFinderMwkSnpt.tokenizeDocuments() - Tokenzing documents, using "
                + MyEnglishAnalyzer.class + " using reflection (yuck). Outputting to: " + tokenizedDocumentsPath);
        System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.tokenizeDocuments() - " + documentsSequencePath + " ===> "
                + tokenizedDocumentsPath);
        DocumentProcessor.tokenizeDocuments(documentsSequencePath, MyEnglishAnalyzer.class, tokenizedDocumentsPath,
                configuration);
        return tokenizedDocumentsPath;
    }

    private static Path writeToSequenceFile(Configuration configuration, Path documentsSequencePath,
            String[] mwkSnippetCategoryDirs) throws IOException {
        System.err.println(
                "SRIDHAR MahoutTermFinderMwkSnpt.main() - Creating sequence file from mwk snippet files, outputting files to sequence file "
                        + documentsSequencePath + " (large)");
        SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(configuration), configuration,
                documentsSequencePath, Text.class, Text.class);

        for (String path : mwkSnippetCategoryDirs) {
            Text cateogoryDir = new Text(Paths.get(path).getFileName().toString());
            DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(Paths.get(path));
            System.err.println(
                    "SRIDHAR MahoutTermFinderMwkSnpt.main() - " + cateogoryDir + "::" + Paths.get(path).toFile());
            try {
                for (java.nio.file.Path fileInPath : stream) {
                    if (Files.isDirectory(fileInPath)) {
                        // listFiles(entry);
                        System.err.println(
                                "SRIDHAR MahoutTermClusterMwkSnpt.writeToSequenceFile() - skipping nested dir: "
                                        + fileInPath);
                    } else {
                        if (fileInPath.toFile().exists()) {
                            // System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writeToSequenceFile() -
                            // writing to sequence file: " + fileInPath);
                            writer.append(cateogoryDir,
                                    new Text(FileUtils.readFileToString(Paths.get(fileInPath.toUri()).toFile())));
                            // TODO: this is wrong, it's overwriting previous files in the same dir
                            // I shouldn't be adding the category anywhere.
                            // https://github.com/technobium/mahout-tfidf
                        }
                    }
                }
            } catch (IOException e3) {
                throw e3;
            } finally {
            }
        }

        writer.close();
        return documentsSequencePath;
    }

    private static class MyEnglishAnalyzer extends StopwordAnalyzerBase {
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
