FILENAME="$@"
#echo "$FILENAME"

# sed 's/.*/s|^&$||g/' ~/sarnobat.git/stopwords.txt > /tmp/stopwords.sed
cat <<EOF > /tmp/stopwords_remove.sh
cat \- | sed -e "$(cat /Volumes/git/github/mahout/shell_simplification/stopwords.sed)"
EOF

cat <<EOF > /tmp/cleansing_custom.sh
	cat \- \
		| perl -pe 's{^\d\d\s*\n}{}g' \
		| grep -v 'dont'
		
EOF

cat <<EOF > /tmp/term_frequencies.sh
cat "$FILENAME" \
	| perl -pe 's{\s+}{\n}g' \
	| perl -pe 's{^[^a-zA-Z]+.*\n}{}g' \
	| tr '[:upper:]' '[:lower:]' | tee /tmp/without_capitals.txt  \
	| tr -d '[:punct:]' | tee /tmp/without_punctuation.txt  \
	| sh /tmp/stopwords_remove.sh  \
	| grep -v '^\s*\$' | tee /tmp/without_blanks.txt \
	| sh /tmp/cleansing_custom.sh \
	| perl -pe 's{^}{$FILENAME: }g' \
	| sort | uniq -c | sort
EOF


sh /tmp/term_frequencies.sh

# We'll have too many scripts if we don't clean them up
#rm /tmp/term_frequencies.sh