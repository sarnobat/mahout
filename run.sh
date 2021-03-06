cd $HOME/github/mahout/mahout-tfidf || echo "no such dir" 
find output/ temp_intermediate  -type f |  xargs --delimiter '\n' --max-args=1 rm
#git update-index --assume-unchanged <file>
# mvn --quiet exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinder" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | tee ../report.txt
#-----------------------------------------------------------------------------------------

#cd mahout-tfidf

##
## Compile
##
echo $PWD
test -e pom.xml || echo "No pom.xml" 
test -e /Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home
JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home
mvn compile  --quiet  --settings ~/sarnobat.git/mac/.m2/settings.xml -Dlogback.configurationFile=$HOME/sarnobat.git/logback.silent.xml

##
## Execute
##

echo "1) mwk files - finding terms (skipped - this one is slow)"
#mvn exec:java --quiet --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwk" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | sort > ../report_mwk.txt
echo "2) mwk snpt files - finding terms"
find ~/sarnobat.git/mwk/snippets/ -type f | shuf | head -30 | mvn exec:java --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwkSnptPiped" -DtermLimitPerFile=5 -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | sort > ../report_mwk_snpts.txt
#mvn exec:java --quiet --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwkSnpt" -Dorg.slf4j.simpleLogger.defaultLogLevel=warning | sort > ../report_mwk_snpts.txt
echo "2) mwk snpt files - clsutering"
find ~/sarnobat.git/mwk/snippets/ -maxdepth 2 -type f | head -320 | mvn exec:java --quiet  --settings ~/sarnobat.git/mac/.m2/settings.xml -Dexec.mainClass="com.technobium.MahoutTermFinderMwkSnptRefactoredCluster" -Ddebug=false -Dthreshold=2.5 -DshowTerms=false | tee ../report_snpts.txt