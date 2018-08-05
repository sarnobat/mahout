cd $HOME/github/mahout/mahout-tfidf || echo "no such dir" 
# mvn --quiet exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinder" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report.txt
#-----------------------------------------------------------------------------------------

#cd mahout-tfidf

##
## Compile
##
echo $PWD
ls pom.xml || echo "No pom.xml" 
mvn compile --quiet  --settings ~/sarnobat.git/mac/.m2/settings.xml -Dlogback.configurationFile=$HOME/sarnobat.git/logback.silent.xml

##
## Execute
##

mvn exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwkSnpt" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report_snpts.txt

