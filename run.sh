cd $HOME/github/mahout/mahout-tfidf || echo "no such dir" 
# mvn --quiet exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinder" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report.txt
#-----------------------------------------------------------------------------------------

#cd mahout-tfidf

##
## Compile
##
echo $PWD
test -e pom.xml || echo "No pom.xml" 
mvn compile --quiet  --settings ~/sarnobat.git/mac/.m2/settings.xml -Dlogback.configurationFile=$HOME/sarnobat.git/logback.silent.xml

##
## Execute
##

echo "1) Finding terms in mwk files (skipped - this one is slow)"
#mvn exec:java --quiet --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwk" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | sort > ../report_mwk.txt
echo "2) Finding terms in mwk snpt files"
mvn exec:java --quiet --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwkSnpt" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | sort > ../report_mwk_snpts.txt

