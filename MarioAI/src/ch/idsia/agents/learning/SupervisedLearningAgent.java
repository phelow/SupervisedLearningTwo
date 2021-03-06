
package ch.idsia.agents.learning;
/*
 * Copyright (c) 2009-2010, Sergey Karakovskiy and Julian Togelius
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Mario AI nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

import weka.core.FastVector;
import ch.idsia.agents.Agent;
import ch.idsia.benchmark.mario.engine.sprites.Mario;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.evolution.Evolvable;
import ch.idsia.evolution.MLP;
import java.util.ArrayList;
import java.util.List;

import ch.idsia.agents.LearningAgent;
import ch.idsia.benchmark.tasks.LearningTask;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.MarioAIOptions;
import edu.stanford.cs229.agents.ActionQtable;
import edu.stanford.cs229.agents.MarioAction;
import edu.stanford.cs229.agents.MarioState;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;
import weka.core.Instance;
import weka.core.*;
import weka.core.Instances;
import weka.classifiers.*;
import weka.filters.unsupervised.instance.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
/**
 * Created by IntelliJ IDEA.
 * User: Sergey Karakovskiy, sergey.karakovskiy@gmail.com
 * Date: Apr 8, 2009
 * Time: 4:03:46 AM
 */

public class SupervisedLearningAgent extends BasicMarioAIAgent implements Agent
{
RandomForest m_randomForest;
boolean m_randomForestEnabled;
int trueJumpCounter = 0;
int trueSpeedCounter = 0;
private MarioState currentState;
private Classifier m_classifier;
String c_marioPerfectRunsDataPath = "";
Instances isTrainingSet = null;
Instance iExample;
FastVector fvWekaAttributes;
String[] tokens;

public enum classifierType{
	multilayerPerceptron,
	naiveBayesSimple,
	KNN,
	decisionTreeJ48,
	decisionRandomForest,
	SMO_RBF,
	SMO_Polynomial
}


public boolean use10foldcross = true;
public classifierType m_ct = classifierType.SMO_Polynomial;

public static void TrainTestSplit(Instances data, Classifier scheme, String name) throws Exception{
	StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
	String[] options = new String[6];
	options[0] = "-N";                  // Indicate we want to set the number of folds
	options[1] = Integer.toString(5);   // Split the data into 5 random folds
	options[2] = "-F";                  // Indicate we want to select a specific fold
	options[3] = Integer.toString(1);   // Select the first fold
	options[4] = "-S";                  // Indicate we want to set the random seed
	options[5] = Integer.toString(1);   // Set the random seed to 1

	filter.setOptions(options);
	filter.setInputFormat(data);
	filter.setInvertSelection(false);
	Instances test = Filter.useFilter(data, filter);
	filter.setInvertSelection(true);
	Instances train = Filter.useFilter(data, filter);
	
	Evaluation eval = new Evaluation(train);
	eval.evaluateModel(scheme,test);
	System.out.println(eval.toSummaryString("\n" + name + " (Train/Test) Results\n======\n",true));
	System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
}

public SupervisedLearningAgent(Instances trainingSet) throws Exception
{
    super("SupervisedLearningAgent");
   isTrainingSet = trainingSet; 
   m_classifier = null;
   isTrainingSet.setClassIndex(12);
   switch(m_ct){
   case multilayerPerceptron:
	    m_classifier = (Classifier) new MultilayerPerceptron();
	    ((MultilayerPerceptron)m_classifier).setHiddenLayers("10");
	    ((MultilayerPerceptron)m_classifier).setTrainingTime(100);
	    ((MultilayerPerceptron)m_classifier).buildClassifier(trainingSet);
		if (use10foldcross) {
			Evaluation eval = new Evaluation(trainingSet);
			eval.crossValidateModel(((MultilayerPerceptron)m_classifier),trainingSet,10,new Random(1));
			System.out.println(eval.toSummaryString("\nMLP (10 Fold Cross) Results\n======\n",true));
			System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		}
		else {
			TrainTestSplit(trainingSet, ((MultilayerPerceptron)m_classifier), "MLP");
		}
	   break;
   case naiveBayesSimple:
	    m_classifier = (Classifier)new NaiveBayes();
	    m_classifier.buildClassifier(trainingSet);
		if (use10foldcross) {
			Evaluation eval = new Evaluation(trainingSet);
			eval.crossValidateModel(m_classifier, trainingSet, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nNaive Bayes Results\n======\n",true));
			System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		}
		else {
			TrainTestSplit(trainingSet, m_classifier, "Naive Bayes");
		}
	   
	    m_classifier = (Classifier) new NaiveBayes();
	   break;
   case KNN:
	   	m_classifier = new IBk();
	   	((IBk )m_classifier).setKNN(1);
	   	((IBk )m_classifier).buildClassifier(trainingSet);
		if (use10foldcross) {
			Evaluation eval = new Evaluation(trainingSet);
			eval.crossValidateModel(m_classifier, trainingSet, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nKNN Results\n======\n",true));
			System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		}
		else {
			TrainTestSplit(trainingSet,m_classifier,"KNN");
		}
	   
	   break;
   case decisionTreeJ48:
	   m_classifier = new J48();
	   ((J48)m_classifier).setConfidenceFactor((float) 0.01);
	   ((J48)m_classifier).buildClassifier(trainingSet);
	   if (use10foldcross) {
		   Evaluation eval = new Evaluation(trainingSet);
		   eval.crossValidateModel(m_classifier, trainingSet, 10, new Random(1));
		   System.out.println(eval.toSummaryString("\nJ48 Results\n======\n",true));
		   System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
	   }
		else
		{
			TrainTestSplit(trainingSet,m_classifier,"J48");
		}
	   break;
   case decisionRandomForest:
	   m_classifier = new RandomForest();
	   m_classifier.buildClassifier(trainingSet);
		if (use10foldcross) {
			Evaluation eval = new Evaluation(trainingSet);
			eval.crossValidateModel(m_classifier, trainingSet, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nRandom Forest Results\n======\n",true));
			System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		}
		else {
			TrainTestSplit(trainingSet,m_classifier,"Random Forest");
		}
	   break;
   case SMO_RBF	:
	   		// Set C in the range from 0.01 to 100.0
			// RBF for gaussian
			// Set gamma from 1 to 10
	   		m_classifier = new SMO();
			RBFKernel rbf = new RBFKernel();
			rbf.setGamma(1);
			((SMO)(m_classifier)).setKernel(rbf);
			((SMO)(m_classifier)).setC(0.01);
			m_classifier.buildClassifier(trainingSet);
			if (use10foldcross) {
				Evaluation eval = new Evaluation(trainingSet);
				eval.crossValidateModel(m_classifier, trainingSet, 10, new Random(1));
				System.out.println(eval.toSummaryString("\nSMO (RBF) Results\n======\n",true));
				System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
			}
			else {
				TrainTestSplit(trainingSet,m_classifier,"SMO (RBF)");
			}
	   break;
   case SMO_Polynomial	:
	// Set C in the range from 0.01 to 100.0
			// Polynomial
			// Set exponent to 1 for linear, 2 for quadratic, and 3 for cubic kernel
			//     (No kernel for dot product)
			PolyKernel pk = new PolyKernel();
			pk.setExponent(1);
			m_classifier = new SMO();
			((SMO)(m_classifier)).setKernel(pk);
			((SMO)(m_classifier)).setC(0.01);
			m_classifier.buildClassifier(trainingSet);
			if (use10foldcross) {
				Evaluation eval = new Evaluation(trainingSet);
				eval.crossValidateModel(m_classifier, trainingSet, 10, new Random(1));
				System.out.println(eval.toSummaryString("\nSMO (Polynomial) Results\n======\n",true));
				System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
			}
			else {
				TrainTestSplit(trainingSet,m_classifier,"SMO (Polynomial)");
			}
	   break;
   
   }
   
    m_classifier.buildClassifier(isTrainingSet);
    
    System.out.println(isTrainingSet.toSummaryString());
    
    currentState = new MarioState();
    
    SetUpWekaInterface();
    reset();
    
  //TODO: fix
	  //TODO: label is not added, not sure if it needs to be
	  fvWekaAttributes = new FastVector(currentState.fields.size()+1);
	  
	  //Add all the numeric attributes
	  for(int i = 0; i <currentState.fields.size(); i++){
		  Attribute a = new Attribute("i" + currentState.fields.get(i).name);
		  fvWekaAttributes.addElement( a);
	  }

	  FastVector fvClassVal = new FastVector(36);
	  String nLabels = GenerateAllLabels(0,"");

	  tokens = nLabels.split(",", -1);
	  for(int i = 0; i < 64; i++){
		  //System.out.println(tokens[i]);
		  fvClassVal.addElement(tokens[i]);
	  }
	  
		 
	  Attribute ClassAttribute = new Attribute("MovementLabel",fvClassVal);
	  
	  fvWekaAttributes.addElement(ClassAttribute);
}


public void SetUpWekaInterface(){
	BufferedReader breader = null;
	//breader = new BufferedReader(new FileReader(c_marioPerfectRunsDataPath));
	
}

public void reset()
{
    action = new boolean[Environment.numberOfKeys];
    action[Mario.KEY_RIGHT] = true;
    action[Mario.KEY_SPEED] = true;
    trueJumpCounter = 0;
    trueSpeedCounter = 0;
}

private boolean DangerOfAny()
{

        if ((getReceptiveFieldCellValue(marioEgoRow + 2, marioEgoCol + 1) == 0 &&
            getReceptiveFieldCellValue(marioEgoRow + 1, marioEgoCol + 1) == 0) ||
            getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 1) != 0 ||
            getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 2) != 0 ||
            getEnemiesCellValue(marioEgoRow, marioEgoCol + 1) != 0 ||
            getEnemiesCellValue(marioEgoRow, marioEgoCol + 2) != 0)
            return true;
        else
            return false;
}


public void SetRandomForest(RandomForest rf){
	m_randomForestEnabled = true;
	m_randomForest = rf;
}


//----code for tracking level params to be printed
final int numberOfOutputs = 6;
final int numberOfInputs = 383;
final int neuronsPerLevel = 50;
private Environment environment;

double[] outputs;

/*final*/
protected byte[][] levelScene;
/*final */
protected byte[][] enemies;
protected byte[][] prevEnemies;
protected byte[][] mergedObservation;

protected float[] marioFloatPos = null;
protected float[] marioPrevFloatPos = null;
protected float[] enemiesFloatPos = null;

protected int[] marioState = null;

protected int marioStatus;
protected int marioMode;
protected boolean isMarioOnGround;
protected boolean isMarioAbleToJump;
protected boolean isMarioAbleToShoot;
protected boolean isMarioCarrying;
protected int getKillsTotal;
protected int getKillsByFire;
protected int getKillsByStomp;
protected int getKillsByShell;
protected int framesPassed = 0;
private boolean firstFrame = true;

public String GenerateAllLabels(int i, String s){
	if(i == 6){
		return s + ",";
	}
	else{
		i++;
		String s1 = "true"+s;
		String s2 = "false"+s;
		return GenerateAllLabels(i,s1) + GenerateAllLabels(i,s2);
	}
}

public void integrateObservation(Environment environment) {
  // Update the current state.
	//System.out.println("integrateObservation2");

    levelScene = environment.getLevelSceneObservationZ(zLevelScene);
    enemies = environment.getEnemiesObservationZ(zLevelEnemies);
    mergedObservation = environment.getMergedObservationZZ(1, 0);

    this.marioFloatPos = environment.getMarioFloatPos();
    this.enemiesFloatPos = environment.getEnemiesFloatPos();
    this.marioState = environment.getMarioState();

    receptiveFieldWidth = environment.getReceptiveFieldWidth();
    receptiveFieldHeight = environment.getReceptiveFieldHeight();

    // It also possible to use direct methods from Environment interface.
    //
    marioStatus = marioState[0];
    marioMode = marioState[1];
    isMarioOnGround = marioState[2] == 1;
    isMarioAbleToJump = marioState[3] == 1;
    isMarioAbleToShoot = marioState[4] == 1;
    isMarioCarrying = marioState[5] == 1;
    getKillsTotal = marioState[6];
    getKillsByFire = marioState[7];
    getKillsByStomp = marioState[8];
    getKillsByShell = marioState[9];
	
  if(framesPassed > 3){
	  currentState.update(environment);


	  iExample = new DenseInstance(currentState.fields.size());
	  System.out.println("");
	  for(int i = 0; i <currentState.fields.size(); i++){
		  System.out.print(currentState.fields.get(i).getInt());
		  System.out.println(fvWekaAttributes.get(i).toString());
		  System.out.println(currentState.fields.get(i).getInt());
		  iExample.setValue(i, (double)currentState.fields.get(i).getInt());
		  //System.out.println(iExample.toString());
		  
	  }
	  System.out.println("");
	  for(int i = 0; i <currentState.fields.size(); i++){
		  System.out.print(iExample.toString(i));
	  }
	  System.out.println("");
	  //TODO: I think we need to set all this up before we run the thing
	  //System.out.println("currentState.fields.size():" + currentState.fields.size());
	  //System.out.println("fvWekaAttributes.size():" + fvWekaAttributes.size());
	  //System.out.println("fvWekaAttributes.elementAt(i):" + fvWekaAttributes.elementAt(0).toString());
	  //System.out.println("currentState.fields.get(i).getInt():" + currentState.fields.get(0).getInt());
	  //System.out.println(iExample.toString());
	  
	  
	  //isTrainingSet.add(iExample);

	  
	  
	  
	  iExample.setDataset(isTrainingSet);
	  //System.out.println("Printing probability");
	  try{
		  double[] fDistribution = m_classifier.distributionForInstance(iExample);
		  int max = 0;
		  for(int i =0; i< 64; i++){
			  if(fDistribution[max] < fDistribution[i]){
				  max = i;
			  }
			  
			  System.out.println(i + ":" + fDistribution[i]);
		  }
		  int start = 0;
		  
		 for(int  i= 0; i < 6; i++){ 
			  
			  //get the moves from the token
			  if(tokens[max].substring(start, start + 5).equals("false")){
				  start += 5;
				  //System.out.println(false);
				  action[i] = false;
			  }
			  else if(tokens[max].substring(start,start+ 4).equals("true")){
				  start += 4;
				  //System.out.println(true);
				  action[i] = true;
			  }else{
				  //System.out.println("ERROR");
			  }
		 }
		  
	  } catch (Exception e){
		  //System.out.println("Could not print likelihoods");
	  }
	  
/*
 	 try{

 	    //TODO: assemble the array of environment values put directly into fastvector
 		 for(int i = 0; i <currentState.fields.size(); i++){
	    		fv.addElement() //TODO: add to fv here
	    	}
	    	 
	    	
 	 }
 	 catch(Exception e){
 		 assert(false);
 	 }
  }else{
 	 try{

	    	 File newFile = new File("E:\\Development\\SchoolWork Spring 2016\\Learning and Advanced Game AI\\SupervisedLearningForMario\\cs229mario\\Data\\trainingData.txt");
	    	 writer = new BufferedWriter(new FileWriter(newFile));
	    	 writer.write("@relation trainingData\r\n");
	    	 for(int i = 0; i <currentState.fields.size();i++){
	    		writer.write("@attribute i" + currentState.fields.get(i).name + " numeric\r\n"); 
	    	 }
	    	 for(int i = 0; i <6;i++){
		    		writer.write("@attribute o" + i + " {true,false}\r\n"); 
	    	 }
	    	 
	    		writer.write("@data\r\n"); 
	    	 writer.close();
 	 }
 	 catch(Exception e){
 		 assert(false);
 	 }*/
  }
  framesPassed++;
}

protected String predictionsToString(FastVector predictions) {
	  StringBuffer sb = new StringBuffer();
	  sb.append(predictions.size()).append(" predictions\n");
	  for (int i = 0; i < predictions.size(); i++) {
	    sb.append(predictions.elementAt(i)).append('\n');
	  }
	  return sb.toString();
}

FastVector m_fv;
public boolean[] getAction()
{
	
    
    return action;
}
}