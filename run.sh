cd $HOME/github/mahout/mahout-tfidf
# mvn --quiet exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinder" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report.txt
#-----------------------------------------------------------------------------------------

#cd mahout-tfidf

##
## Compile
##
mvn --quiet compile  --settings ~/sarnobat.git/mac/.m2/settings.xml -Dlogback.configurationFile=$HOME/sarnobat.git/logback.silent.xml

##
## Execute
##

mvn --quiet exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderSnpt" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report_snpts.txt

