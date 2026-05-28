(* ::Package:: *)
(*
  Wolfram Language reproduction of the official Harris & Nepomuceno PMX notebook.

  This script keeps the Mathematica/Wolfram NetTrain backend, so it is the closest
  way to reproduce the authors' notebook outside the interactive .nb file.

  Example:
    wolframscript -file run_official_wolfram.wl 2 20 400000 false "1,2,3,4,5"

  Arguments:
    1 partition: 1 or 2
    2 nmols/net: 10, 15, or 20
    3 nepochs: e.g. 400000
    4 includeIp: true or false
    5 seeds: comma-separated list, e.g. "1,2,3,4,5"
*)

scriptDir = DirectoryName[$InputFileName];
SetDirectory[scriptDir];

partition = If[Length[$ScriptCommandLine] >= 2, ToExpression[$ScriptCommandLine[[2]]], 2];
nmols = If[Length[$ScriptCommandLine] >= 3, ToExpression[$ScriptCommandLine[[3]]], 20];
nepochs = If[Length[$ScriptCommandLine] >= 4, ToExpression[$ScriptCommandLine[[4]]], 400000];
includeIp = If[Length[$ScriptCommandLine] >= 5, ToExpression[ToUpperCase[$ScriptCommandLine[[5]]]], False];
seeds = If[Length[$ScriptCommandLine] >= 6, ToExpression /@ StringSplit[$ScriptCommandLine[[6]], ","], {1}];

metaRaw = Import[FileNameJoin[{"data", "molecules_table1.csv"}], "CSV"];
headers = First[metaRaw];
rows = Rest[metaRaw];
meta = AssociationThread[headers -> #] & /@ rows;

rawDir = FileNameJoin[{"data", "converted_interped_official"}];
resultsRoot = FileNameJoin[{"results", "wolfram_partition" <> ToString[partition] <> "_Net" <> ToString[nmols] <> If[includeIp, "Ip", ""] <> "_seeds" <> StringRiffle[ToString /@ seeds, "-"]}];
If[! DirectoryQ[resultsRoot], CreateDirectory[resultsRoot, CreateIntermediateDirectories -> True]];

boolTrue[x_] := ToString[x] == "True" || ToString[x] == "true";
getName[row_] := ToString[row["molecule"]];
getInput[row_] := If[includeIp,
  ToExpression /@ row[[{"C", "H", "N", "O", "Ip_eV"}]],
  ToExpression /@ row[[{"C", "H", "N", "O"}]]
];
getCurve[row_] := Import[FileNameJoin[{rawDir, ToString[row["official_file"]]}], "Table"][[All, 2]];
energytable = Import[FileNameJoin[{rawDir, ToString[First[meta]["official_file"]]}], "Table"][[All, 1]];

scale[x_, min_, max_] := Abs[0.05 + ((x - min)/(max - min))*0.9];
unscale[y_, min_, max_] := Abs[((y - 0.05)*(max - min)/0.9) + min];

articleTable2 = <|
  "p1_noip" -> <|"Propanone" -> 14., "2-Methylpropanal" -> 4., "Hexan-3-one" -> 7., "3,3-Dimethylbutan-2-one" -> 5., "Methanol" -> 12.|>,
  "p1_ip" -> <|"Propanone" -> 16., "2-Methylpropanal" -> 5., "Hexan-3-one" -> 7., "3,3-Dimethylbutan-2-one" -> 6., "Methanol" -> 9.|>,
  "p2_noip" -> <|"Ethanal" -> 13., "Ethanol" -> 26., "Propanal" -> 6., "3-Methylbutan-2-one" -> 13., "Molecular Nitrogen" -> 30.|>,
  "p2_ip" -> <|"Ethanal" -> 14., "Ethanol" -> 23., "Propanal" -> 8., "3-Methylbutan-2-one" -> 12., "Molecular Nitrogen" -> 1940.|>
|>;
articleKey = "p" <> ToString[partition] <> If[includeIp, "_ip", "_noip"];

For[sidx = 1, sidx <= Length[seeds], sidx++,
  seed = seeds[[sidx]];
  SeedRandom[seed];
  testCol = "partition" <> ToString[partition] <> "_test";
  testingRows = Select[meta, boolTrue[#[testCol]] &];
  trainingPoolRows = Select[meta, ! boolTrue[#[testCol]] &];
  traindataRows = RandomSample[trainingPoolRows, nmols];
  input = getInput /@ traindataRows;
  output = getCurve /@ traindataRows;
  inputnames = getName /@ traindataRows;

  {inmin, inmax} = MinMax[input];
  {outmin, outmax} = MinMax[output];
  scaledinput = scale[#, inmin, inmax] & /@ input;
  scaledoutput = scale[#, outmin, outmax] & /@ output;
  training = Thread[Rule[scaledinput, scaledoutput]];

  mynet = NetChain[{
      ElementwiseLayer[LogisticSigmoid],
      LinearLayer[Floor[nmols/3], "Input" -> Length[First[input]]],
      ElementwiseLayer[LogisticSigmoid],
      LinearLayer[Length[energytable]]
    }];

  Print["Training seed ", seed, ", partition ", partition, ", Net", nmols, If[includeIp, "Ip", ""], ", epochs=", nepochs];
  result = NetTrain[mynet, training, All, MaxTrainingRounds -> nepochs];
  trainednet = result["TrainedNet"];

  seedDir = FileNameJoin[{resultsRoot, "seed_" <> ToString[seed]}];
  If[! DirectoryQ[seedDir], CreateDirectory[seedDir]];
  Export[FileNameJoin[{seedDir, "training_molecules.csv"}], Transpose[{inputnames}], "CSV"];

  allPredRows = {"seed", "molecule", "energy_eV", "actual_sigma_a0_2", "pred_sigma_a0_2", "percent_difference"};
  metricRows = {"seed", "molecule", "mape_percent", "max_percent_difference", "energy_at_max_error_eV", "article_table2_max_percent"};

  For[i = 1, i <= Length[testingRows], i++,
    row = testingRows[[i]];
    mol = getName[row];
    testin = getInput[row];
    testout = getCurve[row];
    scaledtestin = scale[testin, inmin, inmax];
    predScaled = trainednet[scaledtestin];
    pred = unscale[predScaled, outmin, outmax];
    pct = Abs[((testout - pred)/testout)*100.0];
    maxpos = First[Ordering[pct, -1]];
    articleVal = If[KeyExistsQ[articleTable2[articleKey], mol], articleTable2[articleKey][mol], Missing[]];
    metricRows = Append[metricRows, {seed, mol, Mean[pct], Max[pct], energytable[[maxpos]], articleVal}];
    allPredRows = Join[allPredRows, Table[{seed, mol, energytable[[j]], testout[[j]], pred[[j]], pct[[j]]}, {j, 1, Length[energytable]}]];
  ];

  Export[FileNameJoin[{seedDir, "predictions_long.csv"}], allPredRows, "CSV"];
  Export[FileNameJoin[{seedDir, "metrics.csv"}], metricRows, "CSV"];
  Print[Grid[metricRows, Frame -> All]];
];

Print["Done. Results saved in: ", resultsRoot];
