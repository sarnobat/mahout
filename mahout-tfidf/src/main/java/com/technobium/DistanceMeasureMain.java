package com.technobium;

import java.util.LinkedList;
import java.util.List;

import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyClusterer;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.ImmutableList;

/**
 * I used this to fix {@link ClusteringDemo} to get useful clustering.
 */
public class DistanceMeasureMain {

	public static void main(String[] args) {
		{
			Vector v1 = toVector("java is very good");
			Vector v2 = toVector("java is very bad");
			double distance = new CosineDistanceMeasure().distance(v1, v2);
			System.out.println("DistanceMeasureMain.main() distance is "
					+ distance);
		}
		{
			Vector v1 = toVector("java is good");
			Vector v2 = toVector("java makes me sick");
			double distance = new CosineDistanceMeasure().distance(v1, v2);
			System.out.println("DistanceMeasureMain.main() distance is "
					+ distance);
		}
		// word order doesn't matter
		{
			Vector v1 = toVector("java is very good");
			Vector v2 = toVector("very bad is java");
			double distance = new CosineDistanceMeasure().distance(v1, v2);
			System.out.println("DistanceMeasureMain.main() distance is "
					+ distance);
		}
		{
			List<Vector> of;
			if (false) {
				Vector v1 = toVector("java is very good");
				Vector v2 = toVector("very bad is java");
				Vector v3 = toVector("java makes me sick");
				of = ImmutableList.of(v1, v2, v3);
			} else {
				Vector v1 = toVector("Atletico Madrid win");
				Vector v2 = toVector("Both apple and orange are fruit");
				Vector v3 = toVector("Both orange and apple are fruit");
				of = ImmutableList.of(v1, v2, v3);
			}
			List<Vector> vectorList = new LinkedList();
			vectorList.addAll(of);
			List<Canopy> canopies = CanopyClusterer.createCanopies(vectorList,
					new CosineDistanceMeasure(), 0.3, 0.3);
			for (Canopy canopy : canopies) {
				System.out.println("DistanceMeasureMain.main() "
						+ canopy.asFormatString());
			}

		}
	}

	private static Vector toVector(String string) {
		String[] words = string.split("\\s");
		Vector v = new SequentialAccessSparseVector(Integer.MAX_VALUE);
		v.set(0, 1.1);
		int i = 0;
		for (String word : words) {
			v.set(Math.abs(word.hashCode()), 1);
		}
		return v;
	}

}
