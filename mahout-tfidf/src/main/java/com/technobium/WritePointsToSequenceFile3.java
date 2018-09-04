package com.technobium;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.StopwordAnalyzerBase;
import org.apache.lucene.util.Version;
import org.apache.mahout.vectorizer.DocumentProcessor;

import com.google.common.base.Preconditions;

/**
 * Write documents to sequence file, with stop word analysis
 */
public class WritePointsToSequenceFile3 {

    public static void main(final String[] args) throws Exception {
        doTermFinding();
    }

    private static void doTermFinding() throws Exception {

        System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog");

        Configuration configuration = new Configuration();
        String tempIntermediate = "temp_intermediate/";

        Path documentsSequencePath1 = writeToSequenceFile(configuration, new Path(tempIntermediate, "sequence"),
                new String[] { System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/aspergers",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/atletico",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/business",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/career",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/equalizer",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/productivity",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/self",
                        System.getProperty("user.home")
                                + "/sarnobat.git/mwk/snippets/self/approval_attention_social_status",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/self/cliquology_and_bullying/",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/soccer",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/tech/programming_tips",
                        System.getProperty("user.home")
                                + "/sarnobat.git/mwk/snippets/tech/programming_tips/functional_programming",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/travel",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/video_editing",
                        System.getProperty("user.home") + "/sarnobat.git/mwk/snippets/wrestling", });
        {

            Map<String, String> sequencesMap = sequenceFileToMap(configuration, documentsSequencePath1);

            for (String sequenceKey : sequencesMap.keySet()) {
                System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() - " + documentsSequencePath1
                        + " - key=" + sequenceKey + ", value=" + sequencesMap.get(sequenceKey).replaceAll("\\n", ""));
            }
            System.out.println(
                    "SRIDHAR MahoutTermClusterMwkSnpt.doTermFinding() -  TOOD: I've done this wrong. I shouldn't be adding the category anywhere. https://github.com/technobium/mahout-tfidf . Skip the TFIDF example and go straight to this better clustering example than the existing one I based on: https://github.com/technobium/mahout-clustering");
        }
        
        {
            {

                Files.deleteIfExists(Paths.get("temp_intermediate/tokenized-documents/part-m-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tf-vectors/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/wordcount/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/df-count/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/tfidf-vectors/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/partial-vectors-0/part-r-00000"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/frequency.file-0"));
                Files.deleteIfExists(Paths.get("temp_intermediate/dictionary.file-0"));

                Files.deleteIfExists(Paths.get("temp_intermediate/tokenized-documents/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tf-vectors/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/wordcount/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/df-count/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/tfidf-vectors/_SUCCESS"));
                Files.deleteIfExists(Paths.get("temp_intermediate/tfidf/partial-vectors-0/_SUCCESS"));
            }
            // No files created so far.
            Path tokenizedDocumentsPath;
            try {
                tokenizedDocumentsPath = tokenizeDocuments(configuration, tempIntermediate, documentsSequencePath1);
            } catch (Exception e) {
                // IllegalStateException could get thrown I think, so we need this
                e.printStackTrace();
                System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.main() - Could not instantiate "
                        + MyEnglishAnalyzer.class + ". Probably there is no public class and constructor.");
                return;
            }
            Preconditions.checkState(Paths.get("temp_intermediate/tokenized-documents/part-m-00000").toFile().exists());
            Preconditions.checkState(Paths.get("temp_intermediate/tokenized-documents/_SUCCESS").toFile().exists());
        }
    }

    private static Map<String, String> sequenceFileToMap(Configuration configuration, Path documentsSequencePath1)
            throws IOException {
        Map<String, String> sequencesMap = new HashMap<String, String>();
        Path path = documentsSequencePath1;
        Configuration conf = configuration;
        FileSystem fs = FileSystem.getLocal(conf);
        SequenceFile.Reader reader = null;
        try {
            reader = new SequenceFile.Reader(fs, path, conf);
            Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
            Writable value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), conf);
            while (reader.next(key, value)) {
                // System.out.println("Key: " + key + " value:" + value);
                sequencesMap.put(key.toString(), value.toString());
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        } finally {
            reader.close();
        }
        return sequencesMap;
    }

    private static Path tokenizeDocuments(Configuration configuration, String outputFolder, Path documentsSequencePath)
            throws IOException, InterruptedException, ClassNotFoundException {
        Path tokenizedDocumentsPath = new Path(outputFolder, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
        System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.tokenizeDocuments() - Adding tokenized documents to folder "
                + tokenizedDocumentsPath);
        System.err.println("MahoutTermFinderMwkSnpt.tokenizeDocuments() - Tokenzing documents, using "
                + MyEnglishAnalyzer.class + " using reflection (yuck). Outputting to: " + tokenizedDocumentsPath);
        System.err.println("SRIDHAR MahoutTermFinderMwkSnpt.tokenizeDocuments() - " + documentsSequencePath + " ===> "
                + tokenizedDocumentsPath);
        DocumentProcessor.tokenizeDocuments(documentsSequencePath, MyEnglishAnalyzer.class, tokenizedDocumentsPath,
                configuration);
        return tokenizedDocumentsPath;
    }

    private static Path writeToSequenceFile(Configuration configuration, Path documentsSequencePath,
            String[] mwkSnippetCategoryDirs) throws IOException {
        System.err.println(
                "SRIDHAR MahoutTermFinderMwkSnpt.main() - Creating sequence file from mwk snippet files, outputting files to sequence file "
                        + documentsSequencePath + " (large)");
        SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(configuration), configuration,
                documentsSequencePath, Text.class, Text.class);

        for (String path : mwkSnippetCategoryDirs) {
            Text cateogoryDir = new Text(Paths.get(path).getFileName().toString());
            DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(Paths.get(path));
            System.err.println(
                    "SRIDHAR MahoutTermFinderMwkSnpt.main() - " + cateogoryDir + "::" + Paths.get(path).toFile());
            try {
                for (java.nio.file.Path fileInPath : stream) {
                    if (Files.isDirectory(fileInPath)) {
                        // listFiles(entry);
                        System.err.println(
                                "SRIDHAR MahoutTermClusterMwkSnpt.writeToSequenceFile() - skipping nested dir: "
                                        + fileInPath);
                    } else {
                        if (fileInPath.toFile().exists()) {
                            // System.out.println("SRIDHAR MahoutTermClusterMwkSnpt.writeToSequenceFile() -
                            // writing to sequence file: " + fileInPath);
                            writer.append(cateogoryDir,
                                    new Text(FileUtils.readFileToString(Paths.get(fileInPath.toUri()).toFile())));
                            // TODO: this is wrong, it's overwriting previous files in the same dir
                            // I shouldn't be adding the category anywhere.
                            // https://github.com/technobium/mahout-tfidf
                        }
                    }
                }
            } catch (IOException e3) {
                throw e3;
            } finally {
            }
        }

        writer.close();
        return documentsSequencePath;
    }

    private static class MyEnglishAnalyzer extends StopwordAnalyzerBase {
        private final CharArraySet stemExclusionSet;

        private static class DefaultSetHolder {
            static final CharArraySet DEFAULT_STOP_SET = StandardAnalyzer.STOP_WORDS_SET;
        }

        public MyEnglishAnalyzer(Version matchVersion) {
            this(matchVersion, DefaultSetHolder.DEFAULT_STOP_SET);
        }

        public MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords) {
            this(matchVersion, stopwords, CharArraySet.EMPTY_SET);
        }

        public MyEnglishAnalyzer(Version matchVersion, CharArraySet stopwords, CharArraySet stemExclusionSet) {
            super(matchVersion, stopwords);
            this.stemExclusionSet = CharArraySet.unmodifiableSet(CharArraySet.copy(matchVersion, stemExclusionSet));
        }

        @Override
        protected TokenStreamComponents createComponents(String fieldName, Reader reader) {
            final Tokenizer source = new StandardTokenizer(matchVersion, reader);
            TokenStream result = new StandardFilter(matchVersion, source);
            // prior to this we get the classic behavior, standardfilter does it for
            // us.
            if (matchVersion.onOrAfter(Version.LUCENE_31))
                result = new EnglishPossessiveFilter(matchVersion, result);
            result = new LowerCaseFilter(matchVersion, result);
            CharArraySet stopwords2;
            try {
                stopwords2 = getStopWords(System.getProperty("user.home") + "/github/mahout/stopwords.txt");
            } catch (IOException e) {
                try {
                    result.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
                e.printStackTrace();
                throw new RuntimeException(e);
            }
            result = new StopFilter(matchVersion, result, stopwords2);
            if (!stemExclusionSet.isEmpty())
                result = new SetKeywordMarkerFilter(result, stemExclusionSet);
            result = new PorterStemFilter(result);
            return new TokenStreamComponents(source, result);
        }

        private static CharArraySet getStopWords(String stoplist) throws IOException {
            List<String> ss = FileUtils.readLines(Paths.get(stoplist).toFile());
            CharArraySet ret = new CharArraySet(Version.LUCENE_CURRENT, ss, false);
            ret.addAll(ss);
            return ret;
        }
    }
}
