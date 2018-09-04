package com.technobium;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
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
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * derived from numeric point clustering. I'm trying to find the code to write
 * my mwk snippets to a sequence file. This also does not contain stop word analysis.
 *
 */
public class WritePointsToSequenceFile2 {

	private static final Logger LOG = LoggerFactory
			.getLogger(MahoutTermFinderMwkSnptRefactored.class);

	public static void main(final String[] args) throws Exception {
		doClustering();
	}

	// @Deprecated // Use {@link MahoutTermClusterMwkSnpt} instead - no, that
	// one is the wrong files.
	private static void doClustering() throws IOException {
		System.err.println("MahoutTermFinderMwkSnptRefactored.doClustering()");

		String BASE_PATH = System.getProperty("user.dir") + "_clustering";
		String POINTS_PATH = BASE_PATH + "/points";
		String CLUSTERS_PATH = BASE_PATH + "/clusters";
		String OUTPUT_PATH = BASE_PATH + "/output";
		java.nio.file.Path clusteringBaseDirPath = Paths.get(BASE_PATH);
		if (clusteringBaseDirPath.toFile().exists()) {
			Files.walkFileTree(clusteringBaseDirPath,
					new SimpleFileVisitor<java.nio.file.Path>() {
						@Override
						public FileVisitResult visitFile(
								java.nio.file.Path file,
								BasicFileAttributes attrs) throws IOException {
							Files.delete(file);
							return FileVisitResult.CONTINUE;
						}

						@Override
						public FileVisitResult postVisitDirectory(
								java.nio.file.Path dir, IOException exc)
								throws IOException {
							Files.delete(dir);
							return FileVisitResult.CONTINUE;
						}
					});
		}

		try {
			System.err.println("SRIDHAR MahoutTermClusterMwkSnpt.start() - ");
			Configuration configuration = new Configuration();

			// Create input directories for data
			final File pointsDir = new File(POINTS_PATH);
			if (!pointsDir.exists()) {
				pointsDir.mkdir();
			}

			// read the point values and generate vectors from input data
			final List<MwkVector> vectors = vectorize(new double[][] {
					{ 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 8, 8 },
					{ 9, 8 }, { 8, 9 }, { 9, 9 } });

			// Write data to sequence hadoop sequence files
			String pointsFile = POINTS_PATH + "/pointsFile";
			Preconditions.checkState(!Paths.get(POINTS_PATH + "/pointsFile")
					.toFile().exists());
			writePointsToFile(configuration, vectors, new Path(pointsFile));
			Preconditions.checkState(Paths.get(POINTS_PATH + "/pointsFile")
					.toFile().exists());

			// Write initial centers for clusters
			int numberOfClusters = 2;
			writeClusterInitialCenters(configuration, vectors, CLUSTERS_PATH,
					numberOfClusters, new Path(CLUSTERS_PATH + "/part-00000"));

			// Run K-means algorithm
			Path inputPointsPath = new Path(POINTS_PATH);
			Preconditions.checkState(Paths.get(POINTS_PATH).toFile().exists());
			Path clustersPath = new Path(CLUSTERS_PATH);
			Preconditions
					.checkState(Paths.get(CLUSTERS_PATH).toFile().exists());
			Path outputPath = new Path(OUTPUT_PATH);
			Preconditions.checkState(!Paths.get(OUTPUT_PATH).toFile().exists());
			HadoopUtil.delete(configuration, outputPath);

			// @param input - the directory pathname for input points
			// * @param clustersIn - the directory pathname for initial &
			// computed clusters
			// * @param output - the directory pathname for output points
			KMeansDriver.run(configuration, inputPointsPath, clustersPath,
					outputPath, 0.001, 10, true, 0, false);
			Preconditions.checkState(Paths.get(OUTPUT_PATH).toFile().exists());

			// Read and print output values
			readAndPrintOutputValues(configuration, new Path(OUTPUT_PATH + "/"
					+ Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000"));
			System.err
					.println("SRIDHAR MahoutTermClusterMwkSnpt.start() - end");
		} catch (final Exception e) {
			LOG.error("MahoutTryIt failed", e);
			e.printStackTrace();
		}

		Files.walkFileTree(clusteringBaseDirPath,
				new SimpleFileVisitor<java.nio.file.Path>() {
					@Override
					public FileVisitResult visitFile(java.nio.file.Path file,
							BasicFileAttributes attrs) throws IOException {
						Files.delete(file);
						return FileVisitResult.CONTINUE;
					}

					@Override
					public FileVisitResult postVisitDirectory(
							java.nio.file.Path dir, IOException exc)
							throws IOException {
						Files.delete(dir);
						return FileVisitResult.CONTINUE;
					}
				});
	}

	private static void writePointsToFile(final Configuration configuration,
			final List<MwkVector> points, Path pointsFile) throws IOException {
		System.err
				.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - begin");
		FileSystem fs = FileSystem.getLocal(configuration);
		System.err
				.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 1");
		SequenceFile.Writer writer = SequenceFile.createWriter(fs,
				configuration, pointsFile, IntWritable.class,
				VectorWritable.class);
		System.err
				.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 2");

		int recNum = 0;
		final VectorWritable vec = new VectorWritable();

		System.err
				.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - 3");
		for (final MwkVector point : points) {
			System.err
					.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - point = "
							+ point);
			vec.set(point.getVector());
			writer.append(new IntWritable(recNum++), vec);
		}
		System.err
				.println("SRIDHAR MahoutTermClusterMwkSnpt.writePointsToFile() - end");
		writer.close();
	}

	private static void writeClusterInitialCenters(
			final Configuration configuration, final List<MwkVector> points,
			String clusterPath, int clusterDesiredCount,
			Path clusterOutputFilePath) throws IOException {
		System.out
				.println("SRIDHAR MahoutTermClusterMwkSnpt.writeClusterInitialCenters() - ");
		final Path writerPath = clusterOutputFilePath;

		FileSystem fs = FileSystem.getLocal(configuration);
		// final Path path = new Path(POINTS_PATH + "/pointsFile");
		final SequenceFile.Writer writer = SequenceFile.createWriter(fs,
				configuration, writerPath, Text.class, Kluster.class);

		for (int i = 0; i < clusterDesiredCount; i++) {
			final MwkVector vec = points.get(i);

			// write the initial centers
			final Kluster cluster = new Kluster(vec.getVector(), i,
					new EuclideanDistanceMeasure());
			writer.append(new Text(cluster.getIdentifier()), cluster);
			System.out
					.println("SRIDHAR MahoutTermClusterMwkSnpt.writeClusterInitialCenters() - cluster = "
							+ cluster.toString());
		}

		writer.close();
	}

	private static void readAndPrintOutputValues(
			final Configuration configuration, Path clusteredPointsInputPath)
			throws IOException {
		FileSystem fs = FileSystem.getLocal(configuration);
		final SequenceFile.Reader reader = new SequenceFile.Reader(fs,
				clusteredPointsInputPath, configuration);

		final IntWritable key = new IntWritable();
		final WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
		int count = 0;
		while (reader.next(key, value)) {
			System.out.printf(
					"SRIDHAR MahoutTermClusterMwkSnpt.readAndPrintOutputValues() - "
							+ "%s belongs to cluster %s\n", value.toString(),
					key.toString());
			LOG.info("{} belongs to cluster {}", value.toString(),
					key.toString());
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
			MwkVector vec = new MwkVector(new RandomAccessSparseVector(
					raw[i].length));
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

		// TODO: this violates Demeter. Fix later once we have it working.
		Vector getVector() {
			return vector;
		}
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
