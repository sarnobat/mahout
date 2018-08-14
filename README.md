# mahout

TODO: Since we're not bothering with Groovy here, you can try creating an Uberjar to avoid missing dependencies.


temp_intermediate/sequence ===> temp_intermediate/tokenized-documents
temp_intermediate/tokenized-documents ===> temp_intermediate/tf-vectors
temp_intermediate/tf-vectors ===> temp_intermediate/tfidf
temp_intermediate/tf-vectors ===> temp_intermediate/tfidf
===> temp_intermediate/dictionary.file-0

## commands

### World counts

hdfs dfs -text output/wordcount/part-r-00000

	2018-08-13 21:06:34,864 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
	albert	1
	apples	1
	arleen	1
	bike	1
	blue	6
	boat	1
	books	1
	bought	1
	car	1
	carpet	1
	coat	1
	dish	1
	do	1
	don	1
	donna	1
	eyes	1
	found	1
	glasses	1
	has	1
	have	1
	i	1
	john	1
	lara	1
	like	1
	likes	1
	marta	1
	mike	1
	need	1
	needs	1
	red	4
	saw	1
	sonia	1
	wants	1
	you	1