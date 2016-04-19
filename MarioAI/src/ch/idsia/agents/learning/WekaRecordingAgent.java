
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

import ch.idsia.agents.Agent;
import ch.idsia.benchmark.mario.engine.sprites.Mario;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.evolution.Evolvable;
import ch.idsia.evolution.MLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;


import src.main.java.weka.classifiers.Classifier;
import src.main.java.weka.classifiers.Evaluation;
import src.main.java.weka.core.Instances;
import src.main.java.weka.filters.Filter;
import src.main.java.weka.filters.supervised.instance.StratifiedRemoveFolds;
import src.main.java.weka.classifiers.functions.MultilayerPerceptron;
import src.main.java.weka.classifiers.bayes.NaiveBayes;
import src.main.java.weka.classifiers.lazy.IBk;
import src.main.java.weka.classifiers.trees.J48;
import src.main.java.weka.classifiers.trees.RandomForest;
import src.main.java.weka.classifiers.functions.Logistic;
import src.main.java.weka.classifiers.functions.SMO;
import src.main.java.weka.classifiers.functions.supportVector.RBFKernel;
import src.main.java.weka.classifiers.functions.supportVector.PolyKernel;
/**
 * Created by IntelliJ IDEA.
 * User: Sergey Karakovskiy, sergey.karakovskiy@gmail.com
 * Date: Apr 8, 2009
 * Time: 4:03:46 AM
 */

public class WekaRecordingAgent extends BasicMarioAIAgent implements Agent
{
int trueJumpCounter = 0;
int trueSpeedCounter = 0;
String c_marioPerfectRunsDataPath = "";
public WekaRecordingAgent()
{
    super("WekaRecordingAgent");
    SetUpWekaInterface();
    reset();
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

public boolean[] getAction()
{
	//Pass the information into the learning algorithm to find the right output
	System.out.println("WekaRecordingAgent.getAction()");
    return action;
}
}