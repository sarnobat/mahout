echo "a) Clustering hello world"
mvn exec:java --quiet --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.ClusteringDemo2" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | sort > ../report_clusters.txt
