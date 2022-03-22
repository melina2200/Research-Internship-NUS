#include <stdint.h>
int32_t bias1[8] = {-84473, -5, -30416, -54539, 1905, 83743, -93665, -59320}; // param2
int32_t bias2[8] = {-31, 31981, -9, 10, -1859, 3044, -118, 34}; // param4
int32_t bias3[16] = {15792,-20695,35137,-1108,-15232,22240,2573,3275,-5492,24338,-380,16100,23198,-7949,-11721,-18592}; // param6
int32_t bias4[16] = {-132,1272,5336,-4591,4034,-17,10683,-800,-15672,-10118,2,77962,1646,945,-12193,-12917}; // param8
int32_t bias5[32] = {26153,18504,5606,-12237,-3959,18929,31051,15268,874,-3018,-25232,9444,43018,-6438,8761,-1328,24001,-18825,9036,25944,9562,-18860,-7128,-10938,5890,6311,-25447,-13608,10738,-13683,-1041,16668}; // param10
int32_t bias6[32] = {2401,1227,-11916,5794,2109,6210,1333,3183,2093,500,2686,3646,9273,-3588,-16499,-3602,-86,2291,1058,700,213,8759,-1162,3181,4773,42093,49826,-2404,1344,-24,1411,76}; // param12
int32_t bias7[32] = {12562,-9461,10519,-4105,-8787,-10862,266,-5068,-7692,-4283,15301,1965,-3654,-2573,-290,6538,4778,-13074,-1880,-2010,5670,10394,-2454,31307,-2396,2355,305,-13996,4364,3718,9911,-11206}; // param14
int32_t bias8[32] = {34411,2553,35610,245,-3738,215,5338,20935,13887,15164,14615,960,226,20467,21384,15345,43434,1142,7304,160,-136,3921,-985,-389,32,304,38594,1443,18400,24923,26975,26754}; // param16
int32_t bias9[64] = {11710,-26333,-18920,18126,34074,16482,4781,-5134,-22041,6073,-80,58455,-17209,-10220,-15613,38857,13937,7705,-16735,38604,-31751,47946,-8735,3782,-15862,36420,63669,6262,-23230,20604,37495,10230,-24261,31843,9523,16385,-12396,23100,686,9905,65675,498,-11637,-31726,4783,-487,24009,-3134,-8602,-14000,2267,20107,28708,-3783,31776,10267,17336,23294,13505,18099,30320,6272,14773,2202}; // param18
int32_t bias10[64] = {-3395,16871,-4608,-1175,9437,20856,707,-1557,-3601,-1750,-131,9231,-1494,15590,-2462,18162,-2229,-1995,15007,-119,-1371,-4522,-1020,758,-1358,-4377,22069,4030,-66,13336,19308,-2486,-1313,17834,15401,-1887,-3686,31245,14737,-5258,-454,4982,-336,18286,14436,18,-628,10193,26,-717,2236,24278,-4203,20418,19128,-4279,-3165,-3975,-2127,14257,-744,-485,-1961,27075}; // param20
int32_t bias11[64] = {-441,22865,5187,-2797,-3036,28146,13534,-18940,-510,11949,-9646,11671,-19139,21481,5688,10610,14684,-6051,18165,-27532,-4902,-8350,33995,11112,-23908,3361,-14995,10470,-21232,14985,549,2180,33893,10715,-1979,12138,-18898,-15837,7183,14697,9566,-27666,5716,9678,-3513,-5483,-4330,-1802,-4180,-16328,-5568,8459,25350,5162,18928,33706,-14174,14082,-20296,21483,23876,24085,30257,6381}; // param22
int32_t bias12[64] = {-1318,30738,-3871,26945,57,35940,-397,-2805,-9897,352,22907,24821,-5157,28086,-69,252,-2031,348,22703,-174,528,-3265,-11438,28699,-7305,29048,-1030,1913,-11,33344,25386,-460,33606,21431,423,32788,1195,15213,20632,25322,-282,-2656,26315,26205,36951,-3230,190,40,-5690,-23,-5283,25862,30001,28064,28951,33802,-73,32354,-900,36956,-723,3961,7095,27592}; // param24
int32_t bias13[128] = {8273,6719,34594,-10595,-17343,20760,-640,20530,-1553,17898,17773,11500,-2227,-11808,-20078,1039,-18820,-9871,-1546,-21457,7922,20352,27292,20307,12930,32939,3619,1149,11101,19384,12885,27086,3941,3864,36306,23457,-34747,2949,42860,11736,-10719,-491,22236,28943,10401,31840,654,-5412,-3346,21523,-8067,-12778,-7333,-25382,35846,19308,-7314,21034,27248,31263,44233,30461,13930,10804,25811,17312,-8624,45510,-638,27631,32696,31101,16116,-16262,-12720,41523,10665,26437,59353,-15013,30965,1895,23889,23355,-24348,352,12033,-2141,22871,1980,9673,-7128,-7925,-6551,-639,2441,32865,17731,3669,5416,43237,-13185,9700,10550,-6199,16882,13756,-10289,14383,714,21879,5073,15912,5716,22802,12842,1257,-10912,25659,-5043,33285,24002,11873,21740,-13578,-3859,7728,3051}; // param26
int32_t bias14[128] = {-5147,-1468,-10776,-2203,-6612,2061,-1591,-16,-2410,24,13655,14147,-3592,-159,-779,-5888,-700,-6676,-1539,-5141,23455,2763,-2119,22083,-818,-2026,378,-3169,-1502,2410,20605,1853,-1753,-7091,-541,-810,-866,-439,-4032,-92,672,-6566,-70,9789,-937,19710,1691,29103,-2879,23611,-2144,-1583,-7065,-3156,19633,4946,-744,-8502,-1862,-14,16795,-896,-819,16817,-3503,24190,-1564,20027,18324,164,-1819,27850,-9572,8072,6070,6059,-3838,17926,21157,241,17051,-4822,17927,27341,3993,23362,-937,146,-2608,-5565,20289,13543,13651,-2668,101,-5269,14999,139,17810,20164,25378,-3313,-7977,-2480,-305,-3005,-492,13577,-3807,-1181,-1090,19883,-243,-2138,3203,-92,-4477,1476,15891,25634,19980,-3,10973,86,18145,20697,24574,-7113}; // param28
int32_t bias15[128] = {4845,2828,5221,-22643,15820,-1950,11693,-1087,30302,5749,-7399,18947,-9779,834,-5202,15991,5543,13960,22202,-13500,12879,12579,23791,13856,-28832,9720,-645,9189,12197,30534,18615,-10050,-13090,-5371,-4415,13561,-7592,-14788,14038,-11549,-5507,-1090,5846,-3276,1363,18,-2011,20340,-26857,22203,5580,-8692,-8714,3047,1187,-5635,-4723,14292,-18714,-6013,6351,-4554,-17878,-2651,-3680,-11299,1743,15245,14875,-331,27449,-8397,-9532,-18455,-4927,4522,-3288,-19607,33458,2876,21753,-3206,-368,-19199,23396,-11083,2273,-11898,33754,16587,-2162,-7305,12541,15155,3896,-15232,-17105,6011,3999,16648,-8735,4814,-3752,17751,8711,-35139,-7406,-856,6182,-23972,-10812,-7923,-21212,16000,8256,-3131,-3566,47726,-6716,13176,27052,4007,12749,-18867,-14784,12443,6635,-5118}; // param30
int32_t bias16[128] = {-3930,-12853,-438,18217,-10034,-9624,21943,607,-8970,-4276,-3770,-5263,-4580,16144,-7264,20204,16751,16286,-164,3306,-1475,17295,13887,17856,-9280,-1383,20917,-10574,-7690,-1885,-1749,-3665,-6618,18668,21300,4730,-281,-1179,-8973,-5002,656,-6638,-3828,20914,-4055,-2535,19999,-3670,4584,2994,-6142,17070,-6766,-3382,21520,-1895,11795,-11530,17853,-4572,-5760,-7915,-331,-5120,-12050,-6818,-5510,13283,1709,-11748,8522,20237,8072,-7611,-6984,-4445,-2790,-5564,25516,-10172,-9395,-1609,19979,-2605,17504,-3242,-10484,2683,6342,-5630,-6461,16024,493,-1466,-1937,-7769,-10557,-7032,24912,1056,-7647,24568,20508,-9419,-2978,-6720,-4517,17668,-10628,15616,22030,562,-6793,-5618,-5848,-9525,22620,-476,-2468,16741,-2775,23614,-1769,-9447,-2068,21715,17608,-3642}; // param32
int32_t bias17[128] = {-6495,-16596,7019,1138,1162,-15656,6209,-5484,9344,14430,-14175,16224,-9727,2382,922,-13436,-10493,-3413,-11207,-8787,-7580,33949,-7151,20271,3085,13630,-19867,7193,-2069,-3386,8670,10215,20189,12767,10073,-4686,14484,11173,-15629,-11950,-661,-517,-12993,11714,6990,-2721,-9082,4040,13491,2165,14467,12562,-12963,6864,6195,-2574,2013,11523,4921,3398,15883,-12381,11329,-16964,-2923,11526,-5728,16325,1004,7016,-15030,20663,22128,-7722,-7903,-22941,-14213,-4202,-10967,2867,-13191,18,-4038,-6088,6489,-18898,1021,-6941,6228,831,16401,8240,-10438,12929,12422,-2269,12274,5709,831,-5645,-10980,10268,-2686,3210,25191,22059,-8641,8152,11854,-17381,4352,-30409,-12701,-31436,16260,-11858,31620,28467,17391,10159,-2269,-4788,15750,-13355,-9909,-24571,-3748,-23789}; // param34
int32_t bias18[128] = {18696,-3076,11421,16170,15983,-5789,-10872,-3255,-7335,-4897,17164,-5655,15355,12900,4312,-6633,-6729,-3935,17467,-4377,2195,13777,-9001,3179,-3172,13751,14294,-4177,-3746,-5385,-10175,-3892,15186,-8827,20292,-2659,11452,-128,-6781,-8992,-12019,-6774,-3877,-7025,-959,-282,-13186,-6523,7356,-10423,1119,1360,864,-6195,12362,-7125,-1814,-15214,-12236,9746,13744,-3771,-13589,-10632,-7168,1119,11117,-12566,-677,-4086,-8491,10834,-6320,19065,13466,10223,-482,-1305,-4645,-6234,-6509,16317,-7302,-7233,-12785,-1248,-1452,14122,-4444,-5772,-516,-12961,-8296,-7376,-8409,-9333,-1056,16270,15173,-15194,-12328,-9472,-5333,8114,1054,14735,2299,14750,-8688,742,-9582,-9031,-3301,18010,12557,-11724,-11647,6305,15069,26006,-4281,-4292,10962,18243,-3270,-10583,-3154,-1320}; // param36
int32_t bias19[128] = {11577,3826,-3238,11852,1432,-12150,1651,7536,8726,-3826,-8333,3364,-9262,-9486,-641,2036,-2700,-6232,8975,4015,-1888,17892,965,-5467,11472,-7177,3278,13155,590,8504,-9240,-715,3877,13446,-3149,8347,-3909,314,1120,-228,14354,-3920,5592,2431,12472,9923,8985,4819,-1219,-5509,-4910,-5121,-8427,2614,7712,8153,-1918,-14302,264,-12695,17470,12316,10558,1350,-6147,8652,-4150,-1653,10734,7885,10814,-1739,-10149,-6805,3105,-2539,1394,-548,6009,-6910,-11658,9457,-6149,-5274,-20222,-9879,-4659,-1903,-790,8175,-735,-17512,7652,7513,-18271,26734,2176,22401,4215,1925,7328,6388,-8256,-377,5445,-10905,10243,-9294,7658,7475,1554,-1865,5150,3749,13344,6672,9935,863,-25271,5331,-905,13485,-10013,13834,5046,14544,-9068,2763}; // param38
int32_t bias20[128] = {12368,-2960,8807,-9416,6580,3541,-1881,343,-11713,-7681,-218,5140,6836,-8834,-4478,-4323,14638,-319,-4326,-4852,1620,14262,-9637,-8154,11982,-9188,11806,-9959,-1986,10763,-4224,-203,351,1664,9864,6462,2701,-8448,-7121,2056,12558,1003,5635,5701,-5332,11451,-3569,-4741,-4971,12935,-12242,690,-11575,-9284,15721,-11102,5586,-4095,-5674,-2448,-2811,-10088,6415,-3373,-6475,8032,-13725,10449,14845,364,9612,-4405,-1724,-3692,9955,11020,-6390,3187,-5574,-5340,-5612,1216,12591,-1760,-9740,913,12924,11479,11680,383,-5389,-6685,12160,2166,8684,3194,1378,11234,-5318,-11505,10072,-3054,10877,11404,11363,-6777,1746,-2000,15,1936,465,14654,2483,-6998,123,-5659,2306,-2781,15876,3459,1204,11748,-7915,11019,-1277,9379,-9962,407}; // param40
int32_t bias21[128] = {-18879,-5435,-10548,8942,1510,16470,-9025,-7270,1989,-9043,-1479,-9673,-7552,-2524,8660,726,14210,14639,-22631,1248,-8408,-3316,-1207,9206,19690,457,-16818,-5793,-5313,8984,5889,-8490,-30578,-8279,9816,-762,-3200,3572,-19864,6063,-8839,-4489,4190,-1396,3053,-8854,-19189,-9846,-16263,-1562,-8162,20888,9014,4480,5400,1780,11570,7911,-348,-5622,-4953,-19186,12829,-27548,-640,-114,10629,6872,9465,-9718,5562,-1326,1442,-6689,-12759,-5424,-4778,-2123,-10486,-8523,-590,-804,12108,12551,7343,-204,6799,8251,-5581,-4676,2890,-21746,6298,1812,3651,6312,-7967,-13998,9587,-2717,-10697,-5090,1041,10006,-11583,-1088,-9021,4614,-1469,-27012,4103,4235,6994,-4751,-9124,-7218,170,20315,-8674,-3877,8753,4742,973,-9728,13576,-8352,12733,16458}; // param42
int32_t bias22[128] = {7158,11430,-6011,7437,-5782,3220,878,-3579,-1979,-2548,-2898,-10100,-7204,-1131,-1853,8988,-1284,-6090,-110,-3075,-505,268,-1424,-1409,-6131,-9301,5389,-9386,-9505,-2294,1257,1777,1325,3453,16127,-2634,-2910,14151,-14159,-1210,11777,2504,-1322,5930,12274,17836,-11895,-7190,7948,-4836,-239,-11918,11009,-306,-3799,-2120,1156,15813,1920,-586,-8720,1188,-16118,-6337,-9200,-7981,11375,250,-10495,17027,-2878,3771,12862,1430,21419,-3801,8674,3634,-7048,-7553,-10998,-5671,7088,-1899,-14605,-9791,-11802,-479,19611,-1960,-649,-12635,4675,-1581,-4128,-10909,-8446,10789,11422,-7417,-12084,-708,-3600,-4942,-6215,-3250,-11155,3457,10736,-5091,3209,486,5656,-6352,4931,13486,1951,11908,-9871,-4063,-1136,-4899,11219,-5839,-584,5917,13481,9589}; // param44
int32_t bias23[128] = {5647,16343,-3027,-10002,18272,3645,8153,3504,1322,-8571,-1015,-15951,-47,2758,-7571,5779,5612,-9114,-2953,1318,-6418,-4574,8364,-3472,-1330,6960,1893,-13923,-12203,1089,8758,19520,-9085,902,-4321,-5397,3214,19609,16362,15552,338,6907,15560,22639,10980,-2730,-14844,-11027,2636,7674,-8485,-3514,-5394,-5386,4594,4176,3158,-16723,8203,9785,-1639,13852,93,-1252,-9570,10219,-8615,-10813,-2191,-5152,-2004,3196,-80,31643,4831,-7187,3354,12997,-9125,14593,-3203,-10075,-10046,382,12609,-3838,-3157,-8300,5557,13470,6147,296,-9291,-348,-6458,-3198,3574,-8096,-3117,2919,9108,-9188,-846,15931,-1434,983,-9081,11260,9425,-5810,3367,-1411,15207,13302,13567,-3089,-9182,-13252,-1537,-6933,13478,5872,-12349,14925,-20543,-2147,10680,12069}; // param46
int32_t bias24[128] = {-12246,16241,-3062,-6491,2592,11244,6073,-3093,11438,-9848,-6920,7022,14109,6416,-7749,-10747,-5165,9667,13207,-6276,8429,227,-2828,10436,10279,4889,4759,-6241,10912,-2503,16108,-5241,-4989,13516,-14409,-1587,10223,14167,12435,2146,-3950,-525,-10910,10145,-7276,-8510,-2021,9368,8107,-7338,-11151,-9627,-1410,16390,-529,-8618,2723,-1017,-4956,3696,-10717,5431,10548,-1407,3229,5585,6993,-12492,-10177,3754,-3978,880,13899,-5849,-1718,-2990,11071,-11884,-3175,7023,-3810,13884,-8167,3747,-5954,3405,-3454,-11264,12822,15049,-2002,-5104,-5763,-6211,11169,2129,-4001,9478,-7788,12269,-8697,-2530,31,-9476,-7856,-11574,-9570,-3442,-2073,4037,6293,8878,-1005,-2970,-1892,2872,-9156,-3120,8980,3101,5858,-1946,11987,7405,-9346,-6761,-15046,9157}; // param48
int32_t bias25[256] = {-17648,-26656,-6395,-9074,-35481,-12865,36994,15838,-17336,-28758,-1967,-1699,-9526,-10721,-27752,792,-20248,-7723,2503,18796,12822,-9042,-15590,-10265,-1098,-4647,-10414,11511,-18409,-22673,4532,33157,1306,-27883,-6558,-17551,-6872,13645,36167,-17555,33117,-43152,-5128,-5351,23875,-11437,10697,16443,21635,41401,-1965,-15158,-10542,-14047,37229,-21485,32376,12562,-36778,-19677,-7602,-12761,-11645,-6323,4697,31082,-34667,42359,-9408,11563,23056,-35611,-23403,14282,-28272,-12258,-2723,-9129,-95,-3100,-41166,-23816,-29878,7185,-47194,-30021,44030,-22054,-3949,-24275,-7240,-32282,-7076,39861,-16105,-12488,-44791,7119,-19898,-12408,-7991,6471,-32522,-21231,-25037,-9898,20255,-26295,-1143,-35026,3245,-17036,-19227,-20924,-51286,-9275,10658,-6232,15293,21255,-18551,3846,-15258,-21994,21904,13851,-18260,-21308,-13714,6654,4720,394,-3534,5303,-4457,37532,-3719,4523,21393,-6842,-26086,-1220,-24899,-20718,-38571,-3872,-5172,-4847,-2387,-13003,-33349,21513,-13442,17839,-18529,-40758,-33966,-18785,-17082,-29958,-8653,24394,-8154,16492,-17120,-4009,-17946,-17709,-8139,-2812,-16804,4375,9415,-4243,-1001,-9223,-6079,11078,5964,-31380,-33810,7836,-14905,3041,-8010,-1772,-24366,-1382,7903,-9208,-1513,-45105,-16710,-4384,27615,-14444,10623,-15672,16443,10915,-1589,-29908,-23300,4050,16710,41565,-14961,-15455,-9490,-4285,-2029,7941,1924,-15518,-15686,14308,-3377,3951,-24928,-33978,2917,6579,-21096,-31229,-17866,-38083,-15776,-34089,17618,-4706,-21753,-16052,2574,-51147,14713,-8584,3479,-18933,-33740,-28449,1948,-9374,9302,-15507,-8662,-11017,-10909,11744,-27427,-21627,28887,19539,-22822,-869,21886,1321}; // param50
int32_t bias26[256] = {6011,-4468,2175,173,15416,3806,11591,1128,1683,3095,-453,3311,-242,-7567,-75,3989,-6278,3382,4199,13248,1850,-1760,-4374,-2724,341,6568,7194,6731,-8346,-1239,-1255,12974,2679,7505,1813,-777,6366,6275,3581,-4095,3663,11252,-300,-817,15640,7811,4833,5425,-5839,9053,10141,2235,10487,2231,2931,4996,-5373,5002,-10712,5399,-2502,6108,-730,-3624,8552,-534,-575,-1560,-3691,-3525,8376,2017,6707,7638,-6237,-2861,8482,8423,7081,7335,3486,1614,6749,-938,-7885,-10325,3593,7455,-3375,-2623,5153,8044,5531,7665,-2224,7840,-721,-4577,8243,3777,-5477,-1603,-8471,-147,8118,-3973,6210,-2886,-2721,8093,7120,6571,3454,4719,10713,3472,-11679,-2616,6649,6786,-6026,-3445,5093,-3201,3264,6112,2120,7410,4606,154,2664,7960,5999,3306,-4782,7727,-4046,-5956,6277,10388,3473,5314,-1149,-4941,-7266,5073,3858,-1897,7385,4888,-8658,-96,5466,-4292,-2938,-12281,-7323,4902,11145,60,-2447,9598,7443,303,5963,-4793,8904,490,5393,-4209,4014,1515,-8475,-685,8872,-10992,6400,10138,-1525,-10936,5396,8564,4305,2166,2024,4381,-3391,-2747,1849,10115,-410,-317,8944,5678,10999,5053,6641,5533,2793,9990,4806,-13327,848,4246,5229,5623,3504,-5914,-1976,-7517,3417,32,7254,12705,-4870,9059,4520,5073,6939,-18393,-8773,7557,2977,4102,-2473,3859,2661,-3580,-244,1537,6478,5461,207,-22236,-257,-4999,-4230,-2445,-12078,5966,-4748,-5615,1671,2458,-8042,-3847,-4973,4096,-13178,-4039,-1835,-2679,-908,-10027,6205,-8440}; // param52
int32_t bias27[256] = {16039,36830,11229,45728,3428,6092,11275,55392,32787,3433,1867,36505,30523,16576,842,37258,15718,46074,47597,20059,24300,-3302,14818,13647,23134,12840,-22635,-1202,36672,-10304,-15940,-4073,17072,18658,39356,11221,1588,21409,4356,35100,36233,3582,28066,-9580,1710,46286,24156,90,-10682,32945,-8570,-13922,1162,18483,12107,18204,34286,22592,5314,15773,24623,46328,44734,28547,386,1321,53308,20494,23614,-26796,19582,17697,47516,15438,-510,30842,28384,-19819,4626,16872,16571,-14702,-24037,-26864,-28542,12107,-2436,18773,8895,34432,6837,4357,32784,3614,46851,-3353,17408,3680,6806,22923,-7725,-12951,-16465,40604,-4801,13278,17325,-10156,46653,13226,-18298,29504,23102,32894,45706,15081,65393,-24557,29146,-9212,1017,-24531,21048,-17804,24048,7091,20163,-17464,22147,28932,2834,32178,-12968,-23397,15574,13957,-7820,-2838,-22213,-1977,5513,14212,42262,40839,1873,3175,-9768,27980,46518,62190,-34365,15546,-11850,6492,-6694,10634,12487,27233,18252,-6113,-443,11980,9612,21642,33517,47723,23527,-39592,-47514,-8866,29102,25522,50078,4575,4671,20862,-1826,12990,-15007,13811,25232,13867,5289,-3504,15480,2818,36838,-26899,17497,-9216,-19413,23231,11911,24510,25555,5043,-13945,36096,3321,26145,-26126,40252,33826,20712,15861,1386,15624,-498,-10383,-1206,30214,7216,-43083,-10357,2705,-5720,24831,-11822,27976,-13309,6785,36107,10032,42829,-10203,29378,-24253,25166,8627,25447,8082,-17184,31730,-974,15953,11794,7858,35991,7863,-12876,5078,31209,34611,33836,17617,3538,3310,17333,-4703,51991,20651,10049,-151,-998,2784,18601}; // param54
int32_t bias29[2] = {27826, -27843}; // param56

