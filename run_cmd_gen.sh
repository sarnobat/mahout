# Forget about using Mahout. Do it manually until you have more training data

# doesn't work well
#cat ~/github/mahout/report_full_3_stopwords_filtered.txt | awk -f ~/github/mahout/mwkSnippetCommands.awk 2>/dev/null  | perl -pe 's{[0-9]{3} (.*).mwk: (.*)}{mv \$(grep -nl "\\b$2" *) $1/}g'

# doesn't work well
#cat ~/github/mahout/report_mwk_snpts.txt | awk -f ~/github/mahout/mwkSnippetCommands.awk 2>/dev/null  | perl -pe 's{[0-9]{2} (.*): (.*)}{mv \$(grep -nl "\\b$2" *) $1/}g'