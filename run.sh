cd mahout-tfidf
mvn --quiet compile -Dlogback.configurationFile=/sarnobat.garagebandbroken/Desktop/sarnobat.git/logback.silent.xml
mvn --quiet -Dlogback.configurationFile=/sarnobat.garagebandbroken/Desktop/sarnobat.git/logback.silent.xml exec:java -Dexec.mainClass="com.technobium.TFIDFTester"