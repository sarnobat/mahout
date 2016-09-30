cd mahout-tfidf
mvn --quiet compile -Dlogback.configurationFile=/sarnobat.garagebandbroken/Desktop/sarnobat.git/logback.silent.xml
mvn --quiet exec:java -Dexec.mainClass="com.technobium.MahoutTermFinder" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report.txt