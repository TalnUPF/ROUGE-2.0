package com.rxnlp.tools.rouge;

import java.io.*;
import java.math.RoundingMode;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;


import com.rxnlp.tools.rouge.ROUGESettings.RougeType;
import com.rxnlp.utils.lang.StopWordsHandler;
import com.rxnlp.utils.lang.WordNetDict;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.tartarus.snowball.SnowballProgram;
import org.tartarus.snowball.ext.EnglishStemmer;

/**
 * 
 * @author Kavita Ganesan
 * www.rxnlp.com
 *
 */
public class ROUGECalculator {

	DecimalFormat df = new DecimalFormat("0.00000");

	private final static Logger log = LogManager.getLogger();
	static MaxentTagger tagger;
	static SnowballProgram stemmer;

	private String POS_SEP = "_";
	private static String SEP = "__";

	WordNetDict wrdnet;
	private StopWordsHandler stopHandler;
	private ROUGESettings settings;

	public static class Result {

		double precision = 0;
		double recall = 0;
		double f = 0;
		int count = 0;
		String name;
		
		public void resetROUGE(){
			precision=0;
			recall=0;
			count=0;
			f=0;
		}

		Result() {};

		Result(Result r)
		{
			this.precision = r.precision;
			this.recall = r.recall;
			this.f = r.f;
			this.count = r.count;
			this.name = r.name;
		}
	}

	private static class DecoratedResult
	{
		final String task;
		final String system;
		final String metric;
		final Result result;

		DecoratedResult(String task, String system, String metric, Result result)
		{
			this.task = task;
			this.system = String.join("\t", system.toUpperCase().split("-"));
			this.metric = metric;
			this.result = new Result(result);
		}
	}

	public static class Task {
		String taskName;
		final Map<String, String> system = new HashMap<>();
		final Map<String, Result> results = new HashMap<>();
		final Map<String, String> reference = new HashMap<>();

		@Override
		public boolean equals(Object obj) {
			Task tf1 = (Task) obj;
			return tf1.taskName.equals(taskName);
		}

		@Override
		public int hashCode() {
			return taskName.hashCode();
		}
	}

	public ROUGECalculator()
	{
		settings = new ROUGESettings();
		SettingsUtil.loadProps(settings);

		if (settings.REMOVE_STOP_WORDS) {
			stopHandler = new StopWordsHandler(settings.STOP_WORDS_FILE);
		}

		if (settings.USE_STEMMER) {
			try {
				Class stemClass = Class.forName("org.tartarus.snowball.ext." + settings.STEMMER);
				stemmer = (SnowballProgram) stemClass.newInstance();
			} catch (ClassNotFoundException e) {
				log.error("Stemmer not found " + e.getMessage());
				log.error("Default englishStemmer will be used");
				stemmer = new EnglishStemmer();
			} catch (InstantiationException e) {
				log.error("Problem instantiating stemmer..." + e.getMessage());
			} catch (IllegalAccessException e) {
				log.error("Illegal Access " + e.getMessage());
			}
		}

		/** load POS tagger only if ROUGE topic or synonyms are true */
		if (settings.ROUGE_TYPE.equals(RougeType.topic) || settings.ROUGE_TYPE.equals(RougeType.topicUniq)
				|| settings.USE_SYNONYMS) {
			tagger = new MaxentTagger(settings.TAGGER_NAME);
			log.info("Loaded...POS tagger.");
		}

		if (settings.USE_SYNONYMS)
		{
			this.wrdnet = new WordNetDict(Paths.get(settings.WORDNET_DIR));
		}
	}
	
	public static void main(String args[]) throws IOException {
		ROUGECalculator rc = new ROUGECalculator();
		PrintWriter resultsWriter = new PrintWriter(new FileWriter(new File(args[0])));

		rc.run(Paths.get(args[1]), p -> true, resultsWriter);
	}

	// run from folder path
	public void run(Path projectDir, Predicate<Path> filter, PrintWriter out) {
		String references = projectDir + "/reference";
		String system = projectDir + "/system";

		Path refPath = Paths.get(references);
		Path sysPath = Paths.get(system);

		try {
			if (Files.exists(refPath) && Files.exists(sysPath)) {
				List<Path> refFiles = Files.list(refPath).filter(filter).collect(Collectors.toList());
				List<Path> sysFiles = Files.list(sysPath).filter(filter).collect(Collectors.toList());
				run(refFiles, sysFiles, out);
			} else {
				log.error("A valid 'references' and 'system' folder not found..Check rouge.properties file.");
				log.error("\nYou need to have a valid 'references' and 'system' folder under " + projectDir
						+ ". Please check your rouge.properties file");
				System.exit(-1);
			}
		} catch (Exception e) {
			log.error(e.getMessage());
			log.error("\nYou need to have a valid 'references' and 'system' folder under " + projectDir
					+ ". Please check your rouge.properties file");
			log.error(e.getMessage());
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public void run(List<Path> refFiles, List<Path> sysFiles, PrintWriter out) throws IOException {
		List<Task> hmEvalTasks = initTasks(refFiles, sysFiles);
		evaluate(hmEvalTasks, out);
	}

	private void evaluate(List<Task> tasks, PrintWriter out) throws IOException {
		List<DecoratedResult> results = new ArrayList<>();

		for (Task task : tasks) {
			for (String ngram : settings.NGRAM) {
				for (String system_name : task.system.keySet()) {

					// get all sentences from the system_name file
					List<String> systemSents = getSentences(task.system.get(system_name));
					List<List<String>> reference_sents = task.reference.entrySet().stream()
							.map(e -> {
								List<String> ref_sents = getSentences(e.getValue());
								if (ref_sents.isEmpty())
									log.warn("No reference sentences");
								return ref_sents;
							})
							.collect(Collectors.toList());

					// the result object for sys file
					Result r = task.results.get(system_name);
					computeRouge(r, ngram, systemSents, reference_sents);
					String str = getROUGEName(settings, ngram);

					// Print results to console
					results.add(new DecoratedResult(task.taskName, r.name.toUpperCase(), str, r));
				}
			}//end of task
		}

		NumberFormat format = NumberFormat.getInstance();
		format.setRoundingMode(RoundingMode.HALF_UP);
		format.setMaximumFractionDigits(2);

		Map<String, List<DecoratedResult>> results_by_system = results.stream()
				.collect(Collectors.groupingBy(r -> r.system + "\t" + r.metric));
		String avg_results = "\nROUGE eval:\nSummary type\tSystem\tROUGE type\tRecall\tPrecision\tF-Score" +
				results_by_system.entrySet().stream()
				.sorted(Map.Entry.comparingByKey())
				.map(e -> {
					List<DecoratedResult> result_set = e.getValue();
					final double avg_precision = result_set.stream()
							.mapToDouble(r -> r.result.precision)
							.average()
							.orElse(0.0);
					final double avg_recall = result_set.stream()
							.mapToDouble(r -> r.result.recall)
							.average()
							.orElse(0.0);
					final double avg_fscore = result_set.stream()
							.mapToDouble(r -> r.result.f)
							.average()
							.orElse(0.0);
//					final int tota_ref_summs = result_set.stream()
//							.mapToInt(r -> r.result.count)
//							.sum();
					return e.getKey()
							+ "\t" + format.format(avg_recall) + "\t" + format.format(avg_precision)
							+ "\t" + format.format(avg_fscore); // + "\t" + tota_ref_summs;
				})
				.collect(Collectors.joining("\n", "\n", "\n"));

		out.write(avg_results);

		String detailed_results = "\n\n\n" + results.stream()
				.map(r -> r.task + "\t" + r.system + "\t" + r.metric + "\t" + format.format(r.result.recall) + "\t" +
						format.format(r.result.precision) + "\t" + format.format(r.result.f))
				.collect(Collectors.joining("\n"));
		out.write(detailed_results);

		out.close();
	}

	private String getROUGEName(ROUGESettings settings, String ngram) {

		StringBuffer sb = new StringBuffer();
		sb.append("ROUGE-" + ngram);

		// TYPE - TOPIC, NORMAL
		if (!settings.ROUGE_TYPE.equals(RougeType.normal))
			sb.append("-").append(settings.ROUGE_TYPE);

		if (settings.REMOVE_STOP_WORDS)
			sb.append("+").append("StopWordRemoval");

		if (settings.USE_STEMMER)
			sb.append("+").append("Stemming");

		if (settings.USE_SYNONYMS)
			sb.append("+").append("Synonyms");

		return sb.toString();
	}

	private List<String> getSentences(String system) {

		// read file into stream, try-with-resources
		List<String> sentList = new ArrayList<>();
		String[] lines = system.split("\\r?\\n");

		Arrays.stream(lines).forEach(line -> {
			line = cleanSent(line);
			sentList.add(line);
		});

		return sentList;
	}

	private String cleanSent(String line) {
		line = line.replaceAll("[0-9]+", " @ ").toLowerCase();
		line = line.replaceAll("(!|\\.|\"|'|;|:|,)", " ").toLowerCase();

		return line;
	}

	private static List<Task> initTasks(List<Path> refFiles, List<Path> sysFiles)
	{
		List<Task> tasks = new ArrayList<>();

		// INITIALIZE SYSTEM FILES FOR EACH TASK
		for (Path sysFile : sysFiles)
		{
			try {
				final String fileName = sysFile.getFileName().toFile().getName();
				final String[] fileToks = fileName.split("_");
				final String system_name = fileToks[1];
				String summary = new String(Files.readAllBytes(sysFile));
				if (fileNamingOK(fileToks, fileName)) {
					Task task = getEvalTask(fileToks, tasks);

					Result r = new Result();
					r.name = system_name;
					task.system.put(system_name, summary);
					task.results.put(system_name, r);
				}
			} catch (Exception e) {
				log.error("Failed to read " + sysFile + ": " + e);
			}
		}

		// INITIALIZE REFERENCE FILES FOR EACH TASK
		for (Path refFile : refFiles)
		{
			try
			{
				final String fileName = refFile.getFileName().toFile().getName();
				final String[] fileToks = fileName.split("_");
				final String ref_name = fileToks[1];
				final String summary = new String(Files.readAllBytes(refFile));

				if (fileNamingOK(fileToks, fileName))
				{
					Task task = getEvalTask(fileToks, tasks);
					task.reference.put(ref_name, summary);
				}
			}
			catch (IOException e) {
				e.printStackTrace();
			}
		}

		return tasks;
	}

	/**
	 * Each unique text, is considered a unique summarization task. Create a
	 * summarization task and add evaluation tasks into tasks
	 * 
	 * @param fileToks
	 * @param tasks
	 * @return
	 */
	private static Task getEvalTask(String[] fileToks, List<Task> tasks) {
		String taskName = fileToks[0].toLowerCase();
		Task task = tasks.stream()
				.filter(t -> t.taskName.equalsIgnoreCase(taskName))
				.findFirst()
				.orElseGet(() -> {
					Task t = new Task();
					t.taskName = taskName;
					return t;
				});
		if (!tasks.contains(task))
			tasks.add(task);
		return task;
	}

	private static boolean fileNamingOK(String[] fileToks, String fileName) {
		if (fileToks.length < 2) {
			log.error("Something seems to be wrong with your file naming convention: " + fileName);
			System.exit(-1);
		}
		return true;
	}

	private void getPOSTagged(List<String> reference, boolean restrictByTopic) {

		List<String> newList = new ArrayList<String>();
		for (String sent : reference) {
			sent = sent.toLowerCase();
			String tagged = tagger.tagString(sent);

			if (restrictByTopic)
				tagged = getRelevantPOS(tagged.toLowerCase());

			if (tagged.trim().length() > 0)
				newList.add(tagged.toLowerCase());
		}

		reference.clear();
		reference.addAll(newList);
	}

	/**
	 * Pick out words that have the relevant POS tags if ROUGE-Topic selected
	 * 
	 * @param tagged
	 * @return
	 */
	private String getRelevantPOS(String tagged) {
		String[] tokens = tagged.split("\\s+");
		StringBuffer b = new StringBuffer();

		for (String t : tokens) {
			if (t.matches(".*(" + settings.SELECTED_TOPICS + ").*")) {
				// b.append(t.split("_")[0]).append(" ");
				b.append(t).append(" ");
			}
		}
		return b.toString().trim();
	}

	void getCombiFor(List<HashSet<String>> li, List<String> daList) {

		for (int i = 0; i < li.size(); i++) {

			ArrayList<String> toRemove = new ArrayList<String>();
			ArrayList<String> toAdd = new ArrayList<String>();

			HashSet<String> hs = li.get(i);

			if (daList.size() > 0) {

				for (String tempStr : daList) {
					for (String s : hs) {
						toRemove.add(tempStr);
						toAdd.add(tempStr + SEP + s);
						// tempStr=tempStr+"$$;"+s;
					}
				}

			} else {
				daList.addAll(hs);
			}

			daList.addAll(toAdd);
			daList.removeAll(toRemove);
		}
	}

	public void computeRouge(Result r, String ngram, List<String> sysSents, List<List<String>> refsSents) {

		r.resetROUGE();

		// For each reference summary
		for (List<String> refSents : refsSents) {

			if (refSents == null || refSents.isEmpty()) {

				log.warn("No reference sentences");

			} else {

				/** ROUGE topic , POS TAG and set n-gram to 1 */
				if (settings.ROUGE_TYPE.equals(RougeType.topic) || settings.ROUGE_TYPE.equals(RougeType.topicUniq)) {
					ngram="1";
					getPOSTagged(refSents, true);
					getPOSTagged(sysSents, true);
				}

				/** ROUGE+Synonyms, POS TAG */
				else if (settings.USE_SYNONYMS) {
					getPOSTagged(refSents, false);
					getPOSTagged(sysSents, false);
				}

				/** Remove Stop Words */
				if (settings.REMOVE_STOP_WORDS) {
					removeStopWords(refSents);
					removeStopWords(sysSents);
				}

				/** Stem Words */
				if (settings.USE_STEMMER) {
					applyStemming(refSents);
					applyStemming(sysSents);
				}

				double overlap = 0;
				double ROUGE = 0;

				refSents.removeIf(String::isEmpty);
				sysSents.removeIf(String::isEmpty);

				Collection<String> hsrefOriginal = getNGramTokens(ngram,refSents);
				Collection<String> refSummaryTokens = getNGramTokens(ngram,refSents);
				Collection<String> sysSummaryTokens = getNGramTokens(ngram,sysSents);

				HashSet<String> theSynonyms = new HashSet<>();

				/*
				 * if(s.LOOSEN_WORD_ORDER){ rearrange(referenceSummaryTokens);
				 * rearrange(systemSummaryTokens); }
				 */

				if (settings.USE_SYNONYMS) {
					removePos(refSummaryTokens);

					// PERFORM COMPARISONS
					if (sysSummaryTokens.size() > 0 && refSummaryTokens.size() > 0) {
						for (String sysTok : sysSummaryTokens) {

							String[] unigramList = sysTok.split(SEP);

							// get list of synonyms for individual words in the
							// system summary
							List<HashSet<String>> listOfSynonyms = new ArrayList<HashSet<String>>();
							for (String unigram : unigramList) {
								theSynonyms = getSynonymList(unigram);
								listOfSynonyms.add(theSynonyms);
							}

							// combine synonyms (combinatorial)
							ArrayList<String> synTokens = new ArrayList<String>();
							getCombiFor(listOfSynonyms, synTokens);

							// for each synonym, check if the reference summary
							// contains it. if at least one matches, then this
							// is a hit
							boolean isMatch = false;

							// now check if the reference summary has any of the
							// synonyms
							for (String syntok : synTokens) {

								if (refSummaryTokens.contains(syntok)) {
									refSummaryTokens.remove(syntok);
									overlap++;
								}
							}
						}
					}
				} else {
					// WE DONT CARE ABOUT SYNONYMS

					for (String sysTok : sysSummaryTokens) {

						if (refSummaryTokens.contains(sysTok)) {
							refSummaryTokens.remove(sysTok);
							overlap++;
						}
					}
				}
				r.count=r.count+1;
				computePrecisionRecall(r,ngram, overlap, hsrefOriginal, sysSummaryTokens, refSents, sysSents);
				
				

			} // SENTS FOUND
		}
		
		finalizePrecisionRecall(r);
	}

	/**
	 * Average the precision and recall and compute f-score
	 * 
	 * @param r
	 */
	private void finalizePrecisionRecall(Result r) {
		// AVERAGE THE P/R
		r.precision = r.precision / r.count;
		r.recall = r.recall / r.count;

		if (r.precision > 0 || r.recall > 0)
			r.f = ((1 + Math.pow(settings.BETA, 2)) * r.recall * r.precision)
					/ ((Math.pow(settings.BETA, 2) * r.precision) + r.recall);

	}

	private void computePrecisionRecall(Result r, String ngram, double overlap, Collection<String> hsrefOriginal,
			Collection<String> sysSummaryTokens, List<String> refSents, List<String> sysSents) {

		// ROUGE-L
		// special case, rouge L, we need to get lcs
		if (ngram.toLowerCase().equals("l")) {

			computeROUGEL(r, refSents, sysSents);
		}

		else
		// EVERYTHING ELSE
		if (overlap > 0) {
			double rougeRecall = overlap / (hsrefOriginal.size());
			double rougePrecision = (overlap) / (sysSummaryTokens.size());
			r.recall = r.recall + rougeRecall;
			r.precision = r.precision + rougePrecision;
		}
	}

	private void computeROUGEL(Result r, List<String> refSents, List<String> sysSents) {
		
		Set<String> unionLCSWords = new HashSet<>();
		Set<String> allRefWords = new HashSet<>();
		Set<String> allSysWords = new HashSet<>();
		

		// Get unique sys summary words
		for (String sys : sysSents) {
			allSysWords.addAll(Arrays.asList(sys.split("\\s+")));
		}

		// Get unique reference summary words
		for (String ref : refSents) {
			allRefWords.addAll(Arrays.asList(ref.split("\\s+")));
		}

		// Get the union LCS
		for (String ref : refSents) {
			// get union of words from reference into a hashset
			for (String sys : sysSents) {
				
				unionLCSWords.addAll(getLCSInternal(ref,sys));
				 
			}
		}
		
		double rougeRecall = unionLCSWords.size() / (double) allRefWords.size();
		double rougePrecision = unionLCSWords.size() / (double) allSysWords.size();
		r.recall = r.recall + rougeRecall;
		r.precision = r.precision + rougePrecision;

	}

	private void rearrange(Collection<String> originalTokens) {
		Collection<String> str = new ArrayList<String>();
		Collection<String> newStr = new ArrayList<String>();
		StringBuffer b = new StringBuffer();

		for (String ref : originalTokens) {
			List<String> theTokens = Arrays.asList(ref.split(SEP));
			Collections.sort(theTokens);

			for (String s : theTokens) {
				b.append(s).append(SEP);
			}
			newStr.add(b.toString());
			b.delete(0, b.length());
		}

		originalTokens.clear();
		originalTokens.addAll(newStr);

	}

	private void removePos(Collection<String> nGrams) {

		List<String> removeList = new ArrayList<String>();
		List<String> addList = new ArrayList<String>();

		for (String theNGram : nGrams) {

			// get individual words
			String[] words = theNGram.split(SEP);
			StringBuffer b = new StringBuffer();

			// for each individual word, get the word without POS tags
			// create n-gram tokens with SEP as the separator
			for (String w : words) {

				String[] wordPOSTokens = w.split(POS_SEP);
				if (wordPOSTokens.length == 2) {
					String wordToken = wordPOSTokens[0];
					String posToken = wordPOSTokens[1];
					b.append(wordToken.trim()).append(SEP);
				}
			}

			addList.add(b.substring(0, b.length() - SEP.length()).toString().trim());
			removeList.add(theNGram);
		}

		nGrams.addAll(addList);// add tokens without the POS tags
		nGrams.removeAll(removeList);// remove tokens with the POS tags. At this
										// point we only want the words and not
										// the POS tags
	}

	private HashSet<String> getSynonymList(String sysTok) {
		String[] toks = sysTok.split(POS_SEP);
		HashSet<String> synonymList = new HashSet<>();

		if (toks[1].contains("nn")) {
			synonymList = wrdnet.getNounSynonyms(toks[0]);
		}

		else if (toks[1].contains("jj")) {
			synonymList = wrdnet.getAdjectiveSynonyms(toks[0]);
		}

		else if (toks[1].contains("vb")) {
			synonymList = wrdnet.getVerbSynonyms(toks[0]);
		}

		// System.err.println(toks[0]+":"+synonymList);
		synonymList.add(toks[0].trim());
		return synonymList;
	}

	private void applyStemming(List<String> sents) {

		List<String> updatedSents = new ArrayList<String>();
		StringBuffer b = new StringBuffer();

		for (String s : sents) {
			b.delete(0, b.length());
			String[] tokens = s.toLowerCase().split("\\s+");

			for (String t : tokens) {

				String[] daToks = t.split("_");
				if (daToks.length > 0)
				{
					// IF NOT STOP WORD KEEP
					stemmer.setCurrent(daToks[0]);

					if (stemmer.stem()) {
						b.append(stemmer.getCurrent());

						if (daToks.length == 2) {
							b.append('_');
							b.append(daToks[1]);
						}
						b.append(" ");
					}
				}
			}

			updatedSents.add(b.toString().trim());
		}

		sents.clear();
		sents.addAll(updatedSents);
	}

	private void removeStopWords(List<String> sents) {

		List<String> updatedSents = new ArrayList<String>();
		StringBuffer b = new StringBuffer();

		for (String s : sents) {
			b.delete(0, b.length());
			String[] tokens = s.toLowerCase().split("\\s+");

			for (String t : tokens) {

				String[] daToks = t.split("_");


				// IF NOT STOP WORD, KEEP
				if (daToks.length<1 || !StopWordsHandler.isStop(daToks[0].trim())) {
					b.append(t).append(" ");
				}
			}
			updatedSents.add(b.toString().trim());
		}

		sents.clear();
		sents.addAll(updatedSents);
	}

	private void printSupportedNgrams() {
		System.out.println("The following n-gram settings are supported:");
		System.out.println("ROUGE-N  (e.g. 1,2,3,..)");
		System.out.println("ROUGE-S  (e.g. S1,S2,S3,..)");
		System.out.println("ROUGE-SU (e.g. SU1,SU2,SU3,..);");
		System.out.println("ROUGE-L  (only option: LCS)");
	}

	/**
	 * Get list of n-grams
	 * 
	 * @param sents
	 * @return
	 */
	private Collection<String> getNGramTokens(String ngram, List<String> sents) {

		Collection<String> ngramList = null;

		if (settings.ROUGE_TYPE.equals(RougeType.topicUniq) || settings.ROUGE_TYPE.equals(RougeType.normalCompressed))
			ngramList = new HashSet<String>();
		else
			ngramList = new ArrayList<String>();

		for (String sent : sents) {

			// ROUGE-1
			if (ngram.equals("1")) {
				String[] tokens = sent.split("\\s+");
				ngramList.addAll(Arrays.asList(tokens));
			}

			else if (ngram.matches("[0-9]+")) {
				int ngramVal = Integer.parseInt(ngram);
				generateNgrams(ngramVal, sent, ngramList);
			}

			else// ROUGE-N
			if (ngram.matches("SU?[0-9]+")) {
				generateSkipGrams(ngram, sent, ngramList);
			}

			else // ROUGE-L
			if (ngram.toLowerCase().equals("l")) {
				// just generate unigrams for now
				generateNgrams(1, sent, ngramList);
			}

			else {
				log.error("Please check your N-GRAM settings. NGRAM=" + settings.NGRAM
						+ " doesn't seem to be supported.");
				printSupportedNgrams();
				System.exit(-1);
			}
		}

		return ngramList;
	}

	private static void generateSkipGrams(String gram, String summary, Collection<String> ngramList) {
		String[] tokens = summary.split("\\s+");

		// split to get the skip-gram size
		String[] gramTokens = gram.split("SU?");

		// is the last part does not contain a number, there is some error
		if (gramTokens.length < 2 && !gramTokens[1].matches("[0-9]+")) {
			log.error("There is an error in your ROUGE-S setting. Please check documentation.");
			System.exit(-1);
		}

		int skip = Integer.parseInt(gramTokens[1]);

		// GENERATE THE SKIP-N-GRAMS
		for (int k = 0; k < tokens.length; k++) {
			String s = tokens[k] + "__";
			int start = k + 1;
			int end = Math.min(k + skip + 1, tokens.length);
			for (int j = start; j < end; j++) {
				String theSkipGram = s + tokens[j] + "__";
				ngramList.add(theSkipGram);
			}

		}

		// if SU specified then also include unigrams
		if (gram.toLowerCase().contains("su")) {
			generateNgrams(1, summary, ngramList);
		}

	}

	private static void generateNgrams(int gram, String summary, Collection<String> ngramList) {
		String[] tokens = summary.split("\\s+");

		// GENERATE THE N-GRAMS
		for (int k = 0; k < (tokens.length - gram + 1); k++) {
			String s = "";
			int start = k;
			int end = k + gram;
			for (int j = start; j < end; j++) {
				s = s + tokens[j] + SEP;
			}

			ngramList.add(s);
		}

	}

	/**
	 * LCS implementation repurposed from: https://www.geeksforgeeks.org/printing-longest-common-subsequence/
	 * @param X - Word array 1
	 * @param Y - Word array 2
	 * @param m - Length of word array 1
	 * @param n - Length of word array 2
	 * @return
	 */
	static List<String> lcs(String [] X, String [] Y, int m, int n) 
	 { 
	     int[][] L = new int[m+1][n+1]; 

	     // Following steps build L[m+1][n+1] in bottom up fashion. Note 
	     // that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]  
	     for (int i=0; i<=m; i++) 
	     { 
	         for (int j=0; j<=n; j++) 
	         { 
	             if (i == 0 || j == 0) 
	                 L[i][j] = 0; 
	             else if (X[i-1].equalsIgnoreCase(Y[j-1])) 
	                 L[i][j] = L[i-1][j-1] + 1; 
	             else
	                 L[i][j] = Math.max(L[i-1][j], L[i][j-1]); 
	         } 
	     } 

	     // Following code is used to print LCS 
	     int index = L[m][n]; 
	     int temp = index; 

	     // Create a character array to store the lcs string 
	     String[] lcs = new String[index+1]; 
	     lcs[index] = ""; // Set the terminating character 

	     // Start from the right-most-bottom-most corner and 
	     // one by one store characters in lcs[] 
	     int i = m, j = n; 
	     while (i > 0 && j > 0) 
	     { 
	         // If current character in X[] and Y are same, then 
	         // current character is part of LCS 
	         if (X[i-1].equalsIgnoreCase(Y[j-1])) 
	         { 
	             // Put current character in result 
	             lcs[index-1] = X[i-1];  
	               
	             // reduce values of i, j and index 
	             i--;  
	             j--;  
	             index--;      
	         } 

	         // If not same, then find the larger of two and 
	         // go in the direction of larger value 
	         else if (L[i-1][j] > L[i][j-1]) 
	             i--; 
	         else
	             j--; 
	     } 

	     ArrayList<String> al=new ArrayList<String>();
	     // Print the lcs 
	     
	     for(int k=0;k<=temp;k++) 
	    	if (!lcs[k].isEmpty()){
	        	al.add(lcs[k]);
	    	}
	     
	     return al;
	 }
  
    /* Utility function to get max of 2 integers */
	public static int max(int a, int b) 
    { 
        return (a > b) ? a : b; 
    } 

	private static List<String> getLCSInternal(String s1, String s2) {
		List<String> lcsWords = new ArrayList<String>();
		int internal_start = 0;

		String[] s1Toks = s1.split("\\s+");
		String[] s2Toks = s2.split("\\s+");

		
		return lcs(s1Toks,s2Toks,s1Toks.length,s2Toks.length);
	}

}
