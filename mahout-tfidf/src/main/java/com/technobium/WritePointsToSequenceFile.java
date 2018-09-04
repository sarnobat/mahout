package com.technobium;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * Extracted from term finder. I'm doing this to try and cluster my real mwk
 * snippets rather than hardcoded single sentences.
 * 
 * But this doesn't have stop word analysis. So check {@link WritePointsToSequenceFile2}
 */
public class WritePointsToSequenceFile {

	private static final Logger LOG = LoggerFactory
			.getLogger(MahoutTermFinderMwkSnptRefactored.class);

	public static void main(final String[] args) throws Exception {
		System.err.println("MahoutTermFinderMwkSnptRefactored.doClustering()");

		String BASE_PATH = System.getProperty("user.dir") + "_clustering";
		String POINTS_PATH = BASE_PATH + "/points";
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
			String string = POINTS_PATH + "/pointsFile";
			Preconditions.checkState(!Paths.get(string).toFile().exists());
			writePointsToFile(new Configuration(), vectors, new Path(string));
			Preconditions.checkState(Paths.get(string).toFile().exists());

			System.err
					.println("SRIDHAR MahoutTermClusterMwkSnpt.start() - end");
		} catch (final Exception e) {
			LOG.error("MahoutTryIt failed", e);
			e.printStackTrace();
		}
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
}
