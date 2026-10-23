# %%%% CELL 0 [markdown] len=231
# RSNA Knee Abnormality Detection

报告弱监督 + 多序列 slot attention 影像模型（生产提交版）。

- 标签：58 个金标 study + 4349 个多语种报告软标签

- 影像：plane x (FS,FatSup) 选 slot，几何排序后每 slot 取切片，分位窗 -> resnet -> 12 头

- FUSE 读取经单线程 cp 落本地 ext4（规避 gcsfuse 多线程 EBADF）


# %%%% CELL 1 [code] len=193
# 0) 环境
import os, sys, warnings
warnings.filterwarnings('ignore')
import torch, numpy, pandas
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('cpus', os.cpu_count())

# %%%% CELL 2 [code] len=30516
# 1) 报告 -> 标签 提取器（多语种，移植自 public notebook）
"""RSNA Knee - 多语种放射科报告 -> 12 标签弱监督提取器

移植自 Kaggle public notebook
`beraterolelk/0-941-sota-rsna-knee-dinov2-radimagenet` (Stage-1 报告提取器)。
原 notebook 的 Stage 1 就是靠它产出绝大部分训练标签。
仅保留报告解析，去掉图像/模型部分。
"""
import re
import unicodedata
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
_PRE = str.maketrans({'ı': 'i', 'İ': 'i', 'I': 'i', 'ß': 'ss', 'đ': 'd', 'Đ': 'd', 'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae'})

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join((ch for ch in text if not unicodedata.combining(ch)))
    text = text.replace('\xad', '')
    text = re.sub('[_\\-/\\\\]+', ' ', text)
    text = re.sub('[ \\t]+', ' ', text)
    return text
_SENT_SPLIT = re.compile('(?<=[.;!?])\\s+|\\n+')

def unwrap(text: str) -> str:
    if not isinstance(text, str):
        return ''
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if out and out[-1] and (not re.search('[.;:!?>*•]$', out[-1])) and (len(out[-1].split()) >= 4) and s and (not s[:1].isupper()):
            out[-1] = out[-1] + ' ' + s
        else:
            out.append(s)
    return '\n'.join(out)

def clauses(text: str):
    norm = normalize(unwrap(text) if FEATURES['unwrap'] else text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]
    merged = []
    for i, c in enumerate(raw):
        if c.endswith(':') and len(c.split()) <= 14 and (i + 1 < len(raw)):
            merged.append(c + ' ' + raw[i + 1])
        merged.append(c)
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend((p.strip() for p in c.split(',') if len(p.split()) > 2))
    return out
FEATURES = {'unwrap': True, 'directional_negation': True, 'oa_inherit': True, 'graded_pathology': True, 'synovitis_backoff': True}

def _rx(*alts: str) -> re.Pattern:
    return re.compile('|'.join(alts))
PRE_NEG = _rx('\\bno\\b', '\\bnot\\b', '\\bwithout\\b', '\\bnegative for\\b', '\\babsence\\b', '\\bno evidence\\b', '\\bfree of\\b', '\\bnone\\b', '\\bneither\\b', '\\bnor\\b', '\\bsin\\b', '\\bno hay\\b', '\\bausencia\\b', '\\bausentes?\\b', '\\bno se\\b', '\\bpas de\\b', '\\bsans\\b', '\\baucune?\\b', '\\bgeen\\b', '\\bzonder\\b', '\\bniet\\b', '\\bkeine?[nmrs]?\\b', '\\bohne\\b', '\\bnicht\\b', '\\bkein\\b', '\\bnema\\b', '\\bbez\\b', '\\bnisu\\b', '\\bnije\\b', '\\bδεν\\b', '\\bχωρις\\b', 'ουδεν', '\\bουτε\\b', '\\bбез\\b', '\\bне\\b', 'липсва', '\\bняма\\b')
POST_NEG = _rx('\\byok\\b', '\\byoktur\\b', 'izlenmemekte', 'saptanmadi', '\\bdegil\\b', 'gozlenmemekte', 'mevcut degil', 'eslik etmiyor', '\\bizlenmedi\\b', 'izlenmemistir', 'saptanmamistir', 'gorulmemistir', '\\bnema znakova\\b', 'bez znakova')
NEGATION = _rx(PRE_NEG.pattern, POST_NEG.pattern, '\\bunremarkable\\b')
NEG_WINDOW = 90

def _negated(clause: str, start: int, end: int) -> bool:
    for m in PRE_NEG.finditer(clause):
        if m.end() <= start and start - m.end() <= NEG_WINDOW:
            if not re.search('\\b(but|however|ancak|fakat|pero|maar|aber|no i|ali|ωστοσο|αλλα|но)\\b', clause[m.end():start]):
                return True
    for m in POST_NEG.finditer(clause):
        if m.start() >= end and m.start() - end <= NEG_WINDOW:
            return True
    return False
NORMALITY = _rx('\\bnormal', '\\bintact\\b', '\\bpreserved\\b', '\\bwithin normal limits\\b', 'limites normales', '\\bconservad', '\\bintegr', '\\bnormales\\b', '\\bdoga(l|ll)\\b', 'korunmus', '\\bnormaldir\\b', 'olagan', '\\buredn', '\\bocuvan', '\\bodrzan', '\\bintakt', '\\bprimjeren', '\\bodrzanog kontinuiteta', '\\bodržan', 'φυσιολογικ', 'ακεραι', 'δεν παρατηρουνται', 'δεν σημειωνονται', 'unauffallig', 'regelrecht', '\\bo\\.?b\\.?\\b', 'нормал', 'запазен', 'съхранен', '\\bбез особености\\b', 'интактн', '\\bgaaf\\b', '\\bnormaal\\b')
NORMAL_PHRASE = _rx('\\bsin alteracion', '\\bsin cambios\\b', '\\bsin particularidad', '\\bsin hallazgos\\b', '\\bsin lesion', '\\bsin signos de (rotura|lesion)', '\\bcontinu[oa]s?\\b', '\\bcontinuidad conservada\\b', '\\bno abnormalit', '\\bno significant abnormalit', '\\bunremarkable\\b', '\\bno evidence of (tear|injury|abnormalit)', '\\bohne auffalligkeit', '\\bkein nachweis\\b', '\\bohne befund\\b', '\\bgeen afwijking', '\\bzonder afwijking', '\\bsans anomalie', "\\bpas d[e']anomalie", '\\bbez osobitosti\\b', '\\bbez znakova (rupture|lezije)\\b', '\\bbez patoloskih\\b', 'χωρις αλλοιωσ', 'χωρις παθολογ', 'δεν παρατηρουνται (αξιολογα|παθολογ)', '\\bбез особености\\b', '\\bбез патологич', '\\bбез данни за\\b', '\\bozel bir ozellik yok', '\\bpatolojik bulgu (yok|izlenmemis)')
UNCERTAIN = _rx('\\bpossible\\b', '\\bprobable\\b', '\\bsuspicious\\b', '\\bsuspected?\\b', 'cannot (be )?exclude', '\\bmay\\b', '\\bquestionable\\b', '\\bequivocal\\b', '\\br/o\\b', '\\bdd\\b', '\\blikely\\b', '\\bsuggest', '\\bcompatible with\\b', '\\bposible\\b', 'sin criterios categoricos', '\\bdudos', '\\bsugier', '\\bmuhtemel\\b', '\\bolasi\\b', '\\bsupheli\\b', '\\bizlenim', '\\bdusundur', '\\bmoguce\\b', '\\bvjerojatno\\b', '\\bsumnja\\b', '\\bmoze odgovarati\\b', 'πιθαν', 'υποπτ', '\\bmoglich', '\\bverdachtig', '\\bfraglich', '\\bv\\.?a\\.?\\b', '\\bwohl\\b', '\\bвъзможно\\b', '\\bвероятно\\b', 'суспект', '\\bmogelijk\\b', '\\bverdacht\\b')

TEAR = _rx('\\btear', '\\btorn\\b', '\\brupture', '\\bdisruption\\b', 'discontinuit', '\\bavuls', '\\bmacerat', '\\bbuckethandle\\b', 'bucket handle', '\\brotura\\b', '\\broturas\\b', '\\bruptura', '\\bdesgarro', '\\broto\\b', '\\bdechirure', '\\bdechire', '\\bscheur', '\\bruptuur', 'gescheurd', '\\briss\\b', 'einriss', '\\bruptur', 'zerreiss', '\\blasion', '\\bausriss', '\\byirtik', '\\byirtig', '\\bkopma\\b', 'butunluk kaybi', '\\brupturu\\b', 'devamsizlik', '\\brupture\\b', '\\bdevamliligi secilememis', '\\bpuknuce', '\\bprekid\\b', '\\bpukotin', '\\bruptur', 'ρηξη', 'ρηξις', 'ρηγμα', 'ασυνεχεια', 'руптура', 'разкъсв', 'разрив', 'скъсв', '\\bлезия\\b')
DEGEN = _rx('degenerat', '\\bmucoid\\b', '\\bmyxoid\\b', '\\bfray', '\\bfissur', 'dejeneratif', '\\bmukoid\\b', 'degenerativn', 'εκφυλ', 'дегенерат', '\\bμυξοειδ', '\\bμυξωδ', '\\bmeniskopat', '\\bmeniscopath', '\\bmuco ?ide\\b', 'aufgefasert', '\\bdejenerasyon\\b')
INJURY = _rx('\\binjur', '\\bsprain', '\\blesion', '\\blasion', '\\bedema\\b', '\\boedema\\b', '\\bodem\\b', '\\bedem\\b', '\\bοιδημα', '\\bодем', '\\bедем', '\\bstrain\\b', '\\bhigh signal\\b', '\\bsignal alteration\\b', '\\bhiperintens', '\\bhyperintens', 'aumento de senal', 'alteracion de senal', 'cambio de senal', '\\bsignalanhebung', '\\bsignalalteration', 'verhoogd signaal', 'sinyal artis', 'αυξημενο σημα', 'повишен сигнал', '\\besguince\\b', '\\bthicken', '\\bzadebljanje\\b', '\\bverdikking\\b', '\\bdistenzij', '\\blaksite\\b', '\\blaxity\\b', '\\bpartial\\b', '\\bparcijaln', '\\bparcial', '\\bpartiel', '\\bpartiell')
_GRADE_RX = re.compile('(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)[\\s:]*(?:grade\\s*)?([1-4]|iv|iii|ii|i)\\b')
_ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}

def _grade_of(clause: str):
    best = None
    for m in _GRADE_RX.finditer(clause):
        v = m.group(1)
        n = _ROMAN.get(v, None) if not v.isdigit() else int(v)
        if n is not None and (best is None or n > best):
            best = n
    return best
ANAT = {'ACL': _rx('anterior cruciate', '\\bacl\\b', 'cruzado anterior', '\\blca\\b', 'croise anterieur', 'voorste kruisband', '\\bvkb\\b', 'vorderes kreuzband', 'vorderen kreuzband', 'vordere kreuzband', 'on capraz', '\\bocb\\b', 'anterior capraz', 'prednji krizni', 'prednjeg krizn', 'προσθι[οα][^ ]* χιαστ', 'προσθιου χιαστου', 'χιαστο[^ ]* συνδεσμ', '\\bχιαστ\\w*', 'предна кръстна', 'предната кръстна', 'предна кръста', 'cruciate ligaments', 'ligamentos cruzados', 'ligaments croises', 'kruisbanden', 'kreuzbander', 'capraz baglar', 'krizn[a-z]* ligament[a-z]*', 'χιαστοι συνδεσμ', 'χιαστων συνδεσμ', 'кръстните връзки', 'кръстни връзки'), 'MCL': _rx('medial collateral', '\\bmcl\\b', 'tibial collateral', 'colateral medial', 'colateral interno', '\\blcm\\b', 'collateral medial', 'collateral interne', 'mediale collaterale', 'binnenband', '\\b(mediale|laterale) banden\\b', '\\bcollaterale banden\\b', 'innenband', 'mediales? kollateral', '\\bic yan bag', 'medial kollateral', '\\biyb\\b', 'medyal kollateral', 'medijalni kolateraln', 'medijalnog kolateraln', 'εσω πλαγι', 'εσωτερικο πλαγι', '\\bπλαγι\\w* συνδεσμ', '\\bπλαγιοι\\b', 'медиален колатерал', 'вътрешна странична', '\\bколатерал\\w*', '\\bcolaterales\\b', '\\bcollateraux\\b', '\\bcollateralen\\b', '\\bkolateralni\\b', 'collateral ligaments', 'ligamentos colaterales', 'ligaments collateraux', 'collaterale banden', 'kollateralbander', 'seitenbander', 'yan baglar', 'kolateraln[a-z]* ligament[a-z]*', 'πλαγιοι συνδεσμ', 'πλαγιων συνδεσμ', 'колатерални връзки', 'страничните връзки'), 'Medial Meniscus': _rx('medial meniscus', '\\bmm\\b(?= tear)', 'medial menisc', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'mediale meniscus', 'binnenmeniscus', 'innenmeniskus', 'medialen? meniskus', 'innenmeniskushinterhorn', 'medyal menisk', '\\bic menisk', 'medijalni meniskus', 'medijalnog meniskusa', 'medijalnom meniskusu', 'medijaln\\w* menisk\\w*', '\\bmedijalnog meniska\\b', 'medijalni menisk', 'εσω μηνισκ', 'μηνισκ[^ ]* του εσω', 'εσω διαμερισμα[^.]{0,40}μηνισκ', 'медиалния менискус', 'медиален менискус', 'вътрешния менискус', 'oba meniska', 'both menisci', 'ambos meniscos', 'beide menisci', 'her iki menisku', 'amfoteroi\\w* mhnisk', 'αμφοτερ\\w* μηνισκ', 'двата менискуса', 'medial (and|&) lateral menisc'), 'Lateral Meniscus': _rx('lateral meniscus', 'lateral menisc', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'laterale meniscus', 'buitenmeniscus', 'aussenmeniskus', 'lateralen? meniskus', 'aussenmeniskushinterhorn', 'lateral menisk', '\\bdis menisk', 'lateralni meniskus', 'lateralnog meniskusa', 'lateralnom meniskusu', 'lateraln\\w* menisk\\w*', '\\blateralnog meniska\\b', 'εξω μηνισκ', 'μηνισκ[^ ]* του εξω', 'εξω διαμερισμα[^.]{0,40}μηνισκ', 'латералния менискус', 'латерален менискус', 'външния менискус', 'oba meniska', 'both menisci', 'ambos meniscos', 'beide menisci', 'her iki menisku', 'αμφοτερ\\w* μηνισκ', 'двата менискуса', 'medial (and|&) lateral menisc')}
OA_EVIDENCE = _rx('osteoarthrit', '\\barthros', '\\bgonarthros', '\\bosteoarthros', 'chondropath', 'chondromalac', 'condropat', 'condromalac', '\\bchondros', '\\bchondrosis\\b', 'chondral (loss|defect|ulcer|thinning|injury|fissur|wear)', 'cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)', '(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage', 'articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)', 'osteophyt', 'osteofit', 'osteofyt', 'osteofito', 'osteophyten', 'spurring', 'joint space narrowing', 'pinzamiento articular', 'reduced joint space', 'kikirdak kayb', 'kikirdak incelme', 'kondropati', 'kondral', 'kikirdak dejener', 'eklem aralig\\w* daral', 'eklem mesafesi daral', 'kikirdak kalinlig\\w* azal', 'kraakbeen', 'gonartrose', 'artrose', '\\bknorpel', 'arthrose', 'gonarthrose', 'hrskavic', 'hondromalac', 'artroz', 'osteoartrit', 'artrotsk', 'artrotick', '\\boa promjen', '\\boa\\b', 'degenerativne promjene hrskav', 'χονδρ[^ ]*παθ', 'αρθριτ', 'αρθρωσ', 'οστεοφυτ', 'χονδρομαλακ', 'αρθρικου χονδρου', 'εξαλειψη του αρθρικου χονδρου', 'διαβρωση του αρθρικου χονδρ', 'λεπτυνση[^.]{0,30}χονδρ', 'φθορα[^.]{0,20}χονδρ', 'артроз', 'хондропат', 'остеофит', 'хрущял[^.]{0,40}(изтън|увред|дефект|липс)', 'изтъняване[^.]{0,30}хрущял', 'хондромалац', 'ulcera[s]? condral', 'cartilago[^.]{0,25}(perdida|adelgaz)', 'icrs grade', 'icrs\\b', 'outerbridge', '\\bdenudation\\b', 'denudacij', 'erozivne promjene', '\\berosion of[^.]{0,20}cartilage', 'kraakbeenlijden', 'kraakbeenverlies')
TF_SITE = _rx('compartment', 'compartimento', 'compartiment', 'kompartman', 'kompartiment', 'kompartment', 'odjelj', 'διαμερισμα', 'компартм', '\\bотдел', 'femorotibial', 'tibiofemoral', 'femoro tibial', 'femorotibiaal', 'femorotibijaln', 'феморотибиал', '\\bft zglob', 'tibiofemoraln', 'condyle', 'condilo', 'kondyl', 'kondil', 'condyl', 'κονδυλ', 'кондил', '\\bplateau', '\\bplato\\b', 'platillo', 'meseta', 'плато', 'tibiaplateau', 'tibijaln\\w* plato', 'tibyal plato', 'tibia plato', 'κνημιαι', 'μηριαι', 'weightbearing', 'weightbaring', 'zona de carga', 'dragende deel', 'agirlik tasiyan', '\\bfemur\\b', '\\btibia\\b', '\\bfemoral\\b', '\\btibial\\b', '\\bfemura\\b', '\\btibije\\b', '\\bmesarthrio\\b', 'μεσαρθριο')
PF_SITE = _rx('patellofemoral', 'femoropatellar', 'femoropatelar', 'patelofemoral', 'retropatellar', 'retrorotulian', 'trochlea', 'troclea', 'troklea', 'trochlear', 'trohlej', 'τροχιλ', '\\bpatella', '\\bpatellar', 'rotulian', '\\brotula\\b', '\\bpatele\\b', 'patellofemoraal', 'femoropatellair', 'επιγονατιδ', 'μηροεπιγονατιδ', 'пател', 'феморопател', 'anterior compartment', 'compartimento anterior', 'prednj\\w* odjeljk', '\\bfp zglob', '\\bpf zglob', '\\bfaset', '\\bfacet', 'patellofemoraln')
SIDE_MEDIAL = _rx('\\bmedial\\w*', '\\bmedyal\\w*', '\\bmedijaln\\w*', '\\bmediaal\\w*', '\\bmediale\\w*', '\\binterno\\b', '\\binterna\\b', '\\binternos\\b', '\\binterne\\b', '\\binnen\\w*', '\\bic\\b', '\\bunutarnj\\w*', '\\bεσω\\w*', '\\bεσωτερικ\\w*', '\\bмедиал\\w*', '\\bвътреш\\w*', '\\bbinnen\\w*', '\\bmediaal\\b', '\\bmediales?\\b')
SIDE_LATERAL = _rx('\\blateral\\w*', '\\bexterno\\b', '\\bexterna\\b', '\\bexternos\\b', '\\bexterne\\b', '\\bdis\\b', '\\blateraln\\w*', '\\baussen\\w*', '\\bbuiten\\w*', '\\bεξω\\w*', '\\bεξωτερικ\\w*', '\\bлатерал\\w*', '\\bвъншн\\w*', '\\bvanjsk\\w*')
SIDE_ANTERIOR = _rx('\\banterior\\w*', '\\bant\\b', '\\bon\\b', '\\bprednj\\w*', '\\bvorder\\w*', '\\bvoorste\\b', '\\bπροσθι\\w*', '\\bпредн\\w*', '\\banteriyor\\w*', '\\bavant\\b', '\\banterieur\\w*')
GLOBAL_OA = _rx('tri ?compartment', 'all three compartment', 'global(ised)? (oa|osteoarthrit)', '\\bgonarthros', '\\bgonartros', '\\bgonarthrose', '\\bgonartrose', 'gonartro', 'goanrtrot', 'gonartrot', 'osteoarthritis of the knee', 'artrosis (de |)(la )?rodilla', 'knee osteoarthrit', '\\bdiz osteoartrit', '\\bgonartroz', 'artroza koljena', 'οστεοαρθριτιδα', 'αρθριτιδα του γονατος', 'εκφυλιστικη οστεοαρθριτ', 'артроза на колянната', 'гонартроз', 'degenerative joint disease', '\\bdjd\\b', 'three compartments', 'compartmens', 'compartments')
DIRECT = {'Effusion': _rx('\\beffusion', 'joint fluid', 'intra ?articular fluid', '\\bhydrops\\b', '\\bhemarthros', '\\bhaemarthros', 'derrame articular', '\\bderrame\\b', 'liquido articular', 'hemartrosis', 'epanchement', 'gewrichtsvocht', '\\bvocht\\b', 'gewrichtseffusie', 'opzetting van suprapatell', 'gelenkerguss', '\\berguss\\b', 'gelenksergu', 'gelenksflussigkeit', 'eklem\\w* ic\\w* sivi', 'efuzyon', 'eklem sivisi', 'eklem mesafesinde sivi', 'sivi (miktari|artisi|birikimi)', 'sivi artis', '\\bsivi\\b[^.]{0,25}artmis', '\\bizljev', '\\bizliv', 'zglobn[^ ]* tekucin', '\\bhidrops\\b', 'αρθρικ[^ ]* υγρ', 'υγρου ενδαρθρικα', 'ενδαρθρικ[^ ]* υγρ', 'ποσοτητα υγρου', 'ενδαρθρικ', 'αρθρικη συλλογη', 'υγρο στην αρθρωση', 'υγρου στην αρθρωση', 'συλλογη υγρου', 'ενθαρθρικ', 'ставен излив', 'излив', 'ставна течност', 'синовиална течност'), 'Synovitis': _rx('synovit', 'sinovit', 'synovial (thickening|proliferation|hypertroph)', 'thicken\\w* synovial', 'hypertroph\\w* of the synovium', 'synoviale? (verdikking|proliferatie)', 'verdikkingen van (het )?synovium', 'synovialitis', 'synovialis(verdickung|proliferation)', 'reizsynovial', 'sinovijalitis', 'sinovitis', 'zadebljanje sinovij', 'proliferacij\\w* sinovij', 'sinovijaln\\w* proliferacij', 'υμενιτιδα', 'συνοβιτιδα', 'υμενικ[^ ]* υπερτροφ', 'αρθρικου υμεν', 'παχυνση[^.]{0,20}υμεν', 'υμενα', 'синовит', 'синовиал[^ ]* (задебел|пролифер)', '\\bpannus\\b', '\\bhoffit', 'sinovyal\\w* (kalinlas|proliferas)', 'sinovyal hipertrof', '\\bartrit\\b', '\\barthritis\\b'), "Baker's": _rx('baker', 'popliteal cyst', 'quiste popliteo', 'quistes popliteos', 'kyste poplite', 'popliteale? cyst', 'poplitealzyste', 'bakerzyste', 'popliteal kist', '\\bbakerova\\b', 'poplitealn[^ ]* cist', 'popliteal\\w* cist', 'κυστη baker', 'πολυχωρη συνοβιακη κυστη', 'κυστη του baker', 'συνοβιακη κυστη', 'κυστη τυπου baker', 'киста на бейкър', 'бейкърова киста', 'поплитеална киста', 'бекеров', 'gastrocnemio ?semimembranos', 'gastrocnemius semimembranosus burs'), 'Contusion': _rx('\\bcontusion', 'bone bruise', 'bone marrow (o?edema|contusion)', 'marrow o?edema', '\\bkontuz', 'medular bone o?edema', 'osseous contusion', 'contusion osea', 'edema oseo', 'edema de medula osea', 'contusiones oseas', 'oedeme osseux', 'contusion osseuse', 'botcontusie', 'botoedeem', 'beenmergoedeem', 'botmergoedeem', 'knochenmarkodem', 'knochenodem', 'knochenmarksodem', 'kontusion', 'kemik kontuzyonu', 'kemik iligi odemi', 'kemik odemi', 'kemik iliginde odem', 'kontuzyonel kemik', 'kemik iligi odemleri', 'kostani edem', 'edem kosti', 'kontuzij', 'kostane srzi[^.]{0,20}edem', 'οστεομυελικ[^ ]* οιδημα', 'οστικο οιδημα', 'μυελικο οιδημα', 'οστικο μωλωπ', 'костномозъчен едем', 'костен едем', 'контузионен', 'костно мозъчен едем'), 'Fracture': _rx('\\bfractur', '\\bfract\\b', '\\bfractura', '\\bfracturas\\b', '\\bfractuur', '\\bbreuk\\b', '\\bfraktur', '\\bbruch\\b', '\\bkirik\\b', '\\bkirigi\\b', '\\bkiri[kg]\\w*', '\\bprijelom', 'impresijsk[^ ]* fraktur', 'impaktcij', 'καταγμα', 'καταγματ', 'фрактур', 'счупван', 'фисур', 'insufficiency fracture', 'stress fracture', 'avulsion fracture', 'subchondral fracture', 'subkondral kiri', 'impaction (fracture|injury)', 'osteochondral (fracture|impaction)', '\\bsegond\\b', 'impactiefractuur', 'subchondrale impression', 'subchondraler? impress')}
DECOY = {'Fracture': _rx('microfractur', '\\bfracture (risk|prophyla)'), "Baker's": _rx('meniscal cyst', 'quiste meniscal', 'parameniscal')}
PAIRED = {'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus'}
OA_TARGETS = ['Medial OA', 'Lateral OA', 'PF OA']
PLURAL_MENISCI = _rx('\\bmenisci\\b', '\\bmeniscos\\b', '\\bmenisques\\b', '\\bmenisken\\b', '\\bmeniskusi\\b', '\\bmenisk\\w*ler\\b', '\\bμηνισκοι\\b', '\\bμηνισκων\\b', '\\bменискуси\\b', '\\bменискусите\\b', '\\bmenisci\\w*\\b')
ANY_SIDE = _rx(SIDE_MEDIAL.pattern, SIDE_LATERAL.pattern)
STEM_MENISCUS = _rx('menisc\\w*', 'menisk\\w*', 'μηνισκ\\w*', 'мениск\\w*')
STEM_CRUCIATE = _rx('cruciate', 'cruzado', 'croise', 'kruisband', 'kreuzband', 'capraz bag\\w*', 'krizn\\w*', 'χιαστ\\w*', 'кръстн\\w*', '\\bacl\\b', '\\blca\\b', '\\bvkb\\b', '\\bocb\\b', '\\bacb\\b')
STEM_COLLATERAL = _rx('collateral\\w*', 'colateral\\w*', 'kollateral\\w*', 'collaterale\\w*', 'kolateraln\\w*', 'yan bag\\w*', 'πλαγι\\w*', 'колатерал\\w*', 'странич\\w*', 'innenband\\w*', 'binnenband\\w*', '\\bmcl\\b', '\\blcm\\b', '\\biyb\\b')
STEM_FRACTURE = _rx('fractur\\w*', 'fraktur\\w*', 'fractuur\\w*', '\\bfract\\b', 'kiri[kgğ]\\w*', 'prijelom\\w*', 'lom kosti', '\\bbreuk\\w*', '\\bbruch\\w*', 'καταγμα\\w*', 'καταγματ\\w*', 'фрактур\\w*', 'счупван\\w*', 'fisur\\w* (osea|oseas|kost)', 'fissur\\w* kost')

def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int=55):
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False
STEM_RULES = {'ACL': (STEM_CRUCIATE, SIDE_ANTERIOR), 'MCL': (STEM_COLLATERAL, SIDE_MEDIAL), 'Medial Meniscus': (STEM_MENISCUS, SIDE_MEDIAL), 'Lateral Meniscus': (STEM_MENISCUS, SIDE_LATERAL)}

class _Matcher:

    def __init__(self, phrase_rx, stem=None, side=None, window=55):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None:
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None
ANAT_MATCH = {t: _Matcher(ANAT[t], *STEM_RULES[t]) for t in PAIRED}
DIRECT_MATCH = {t: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if t == 'Fracture' else rx) for t, rx in DIRECT.items()}
SEV_LOW = _rx('\\bsmall\\b', '\\bminimal\\b', '\\btrace\\b', '\\bmild\\b', '\\bslight\\b', '\\btiny\\b', '\\bscant\\b', '\\bdiscrete\\b', '\\blow ?grade\\b', '\\bincipient\\b', '\\bleve\\b', '\\bminim', '\\bpeque', '\\bfina\\b', '\\bfino\\b', '\\bligero\\b', '\\bescaso\\b', '\\bdiscreto\\b', '\\bhafif\\b', '\\baz miktarda\\b', '\\bsilik\\b', '\\bmanj\\w*', '\\bblago\\b', '\\bdiskretn', '\\bmalo\\b', '\\bpocetn', '\\bgering', '\\bdiskret', '\\bkleine?r?\\b', '\\bwenig\\b', '\\bzarte?\\b', '\\bbeperkte?\\b', '\\bgeringe\\b', '\\bweinig\\b', '\\blichte?\\b', '\\blicht\\b', '\\bηπι', '\\bμικρ', '\\bελαχιστ', '\\bαρχομεν', '\\bминимал', '\\bлек', '\\bмалк', '\\bнеголям')
SEV_HIGH = _rx('\\blarge\\b', '\\bmarked\\b', '\\bmassive\\b', '\\bsevere\\b', '\\bextensive\\b', '\\bmoderate\\b', '\\bgross\\b', '\\bsignificant\\b', '\\babundant\\b', '\\btense\\b', '\\bcomplete\\b', '\\bfull ?thickness\\b', '\\bhigh ?grade\\b', '\\badvanced\\b', '\\bmoderad', '\\bimportante\\b', '\\bsevera?\\b', '\\bmarcad', '\\bcuantios', '\\bespesor total\\b', '\\bcompleta?\\b', '\\bbelirgin\\b', '\\byaygin\\b', '\\bileri\\b', '\\bciddi\\b', '\\bbol\\b', '\\bkomplet', '\\bopsezan\\b', '\\bveliki\\b', '\\bizrazit', '\\bznacajn', '\\bumjeren', '\\buznapredoval', '\\bpotpun', '\\bkompleksn', '\\bausgepragt', '\\bdeutlich', '\\bmassiv', '\\bmassig', '\\bgross', '\\buitgebreid', '\\bgevorderd', '\\bveel\\b', '\\bmatige?\\b', '\\bvolledig', '\\bμετρι', '\\bμεγαλ', '\\bεκτεταμεν', '\\bευμεγεθ', '\\bσοβαρ', '\\bπληρη', '\\bголям', '\\bизразен', '\\bзначим', '\\bумерен', '\\bобилен', '\\bпълн')
GRADE_HIGH = re.compile('grade?[ao]?\\s*(3|4|iii|iv)\\b|icrs grade (iii|iv|3|4)|stupnja iv|stupnja iii|\\bgrado (3|4)\\b|\\bgrad (3|4)\\b|\\bgrade (3|4)\\b')
DEGENERATIVE_MARROW = _rx('subchondral', 'subcondral', 'subkondral', 'supkondraln', 'subchondraln', 'υποχονδρι', 'υπαρθρικ', 'субхондрал', 'subchondrale?', 'subartikuler', '\\bcyst', '\\bquist', '\\bzyste\\b', '\\bcistic', 'reactive', 'reactivo', 'degenerative', 'degenerativ', 'reaktiv', '\\bcisti\\b')
TRAUMA = _rx('\\bbruise\\b', '\\bcontusion', '\\bkontuz', '\\btrauma', '\\bimpaction\\b', '\\bpivot shift\\b', '\\bkissing\\b', '\\bacute\\b', '\\bagudo\\b', '\\bakut', '\\bpivot kaymasi\\b', '\\bcontusion osseuse\\b', '\\bbone bruise\\b', '\\bbotcontusie\\b', '\\bконтузион', '\\bμωλωπ', '\\bkontuzij', '\\bimpaktcij', '\\bimpakcij', '\\bfall\\b', '\\binjury\\b', '\\bimpression\\b')
SYNOVIAL_PROXY = _rx('bursit', 'burzit', '\\bbursa\\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)', 'suprapatellar (bursitis|effusion|recess)', 'suprapatellar bursa', 'suprapatellar bursada', 'suprapatelarno', 'suprapatellaire recessus', 'hoffa', 'hoffit', 'plica', 'plika', 'πλικα', 'fat pad[^.]{0,20}(edema|oedema)', 'kapsul', 'capsul', 'καψ', 'капсул', '\\bpannus\\b', '\\bsinov', '\\bsynov')

def _polarity(clause: str, span=None) -> str:
    if UNCERTAIN.search(clause):
        return 'uncertain'
    if span is None or not FEATURES['directional_negation']:
        if NEGATION.search(clause):
            return 'negative'
    elif _negated(clause, span[0], span[1]):
        return 'negative'
    if NORMALITY.search(clause):
        if TEAR.search(clause) or GRADE_HIGH.search(clause):
            return 'positive'
        return 'negative'
    return 'positive'

def _severity(clause: str) -> float:
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and (not low):
        return 1.0
    if low and (not high):
        return 0.45
    if high and low:
        return 0.8
    return 0.75

def _grade(n_pos, n_neg, n_unc, best):
    if n_pos or n_unc:
        score = min(0.97, 0.5 + 0.45 * best + 0.015 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.2 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = (0.28, 0.05)
    return (score, conf)

def _paired_weight(clause: str, meniscus: bool) -> float:
    g = _grade_of(clause) if FEATURES['graded_pathology'] else None
    tear = TEAR.search(clause) is not None
    if meniscus:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.95 if g >= 3 else 0.3
        elif DEGEN.search(clause):
            base = 0.35
        else:
            base = 0.45
    elif tear:
        base = 1.0
    elif g is not None:
        base = 0.85 if g >= 2 else 0.3
    elif DEGEN.search(clause):
        base = 0.4
    else:
        base = 0.55
    if SEV_HIGH.search(clause) and (not SEV_LOW.search(clause)):
        base = min(1.0, base * 1.2)
    elif SEV_LOW.search(clause) and (not SEV_HIGH.search(clause)):
        base *= 0.7
    return base

def _score_paired(cls, tgt):
    anat_rx = ANAT_MATCH[tgt]
    path_rx = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)
    meniscus = 'Meniscus' in tgt
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        hit = anat_rx.search(c)
        if hit is None and meniscus and PLURAL_MENISCI.search(c) and (not ANY_SIDE.search(c)):
            hit = PLURAL_MENISCI.search(c)
        if hit is None:
            continue
        pm = path_rx.search(c)
        if pm is None and _grade_of(c) is None:
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c) and (not NEGATION.search(c))):
                n_neg += 1
            continue
        span = (pm.start(), pm.end()) if pm is not None else None
        pol = _polarity(c, span)
        if pol == 'positive':
            n_pos += 1
            best = max(best, _paired_weight(c, meniscus))
        elif pol == 'negative':
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.45 * _paired_weight(c, meniscus))
    s, cf = _grade(n_pos, n_neg, n_unc, best)
    return (s, cf, n_pos, n_neg)

def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None, context_bonus=None):
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and (not path_rx.search(c)):
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c) and (not NEGATION.search(c))):
                n_neg += 1
            continue
        pol = _polarity(c, (m.start(), m.end()))
        if pol == 'positive':
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == 'negative':
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.3)
    s, c = _grade(n_pos, n_neg, n_unc, best)
    return (s, c, n_pos, n_neg)

def _score_oa(cls):
    acc = {t: {'pos': 0, 'neg': 0, 'unc': 0, 'best': 0.0} for t in OA_TARGETS}
    g_pos, g_neg, g_best = (0, 0, 0.0)
    for c in cls:
        m = OA_EVIDENCE.search(c)
        if not m:
            continue
        pol = _polarity(c, (m.start(), m.end()))
        sev = _severity(c)
        tf_med = _near(c, TF_SITE, SIDE_MEDIAL, 45)
        tf_lat = _near(c, TF_SITE, SIDE_LATERAL, 45)
        pf = PF_SITE.search(c) is not None
        hits = []
        if tf_med:
            hits.append('Medial OA')
        if tf_lat:
            hits.append('Lateral OA')
        if pf:
            hits.append('PF OA')
        if not hits:
            if pol == 'positive':
                g_pos += 1
                g_best = max(g_best, sev if GLOBAL_OA.search(c) else sev * 0.7)
            elif pol == 'negative':
                g_neg += 1
            continue
        for t in hits:
            if pol == 'positive':
                acc[t]['pos'] += 1
                acc[t]['best'] = max(acc[t]['best'], sev)
            elif pol == 'negative':
                acc[t]['neg'] += 1
            else:
                acc[t]['unc'] += 1
                acc[t]['best'] = max(acc[t]['best'], 0.3)
    out = {}
    for t in OA_TARGETS:
        a = acc[t]
        pos, neg, unc, best = (a['pos'], a['neg'], a['unc'], a['best'])
        if not (pos or unc) and g_pos and FEATURES['oa_inherit']:
            if neg:
                score, conf = _grade(0, neg, 0, 0.0)
                score = max(score, 0.35)
                conf *= 0.7
            else:
                score, conf = _grade(g_pos, 0, 0, g_best * 0.92)
                conf *= 0.75
        else:
            score, conf = _grade(pos, neg + g_neg, unc, best)
        out[t] = (score, conf, pos, neg)
    return out

def extract(report: str) -> dict:
    cls = clauses(report)
    out = {}
    for tgt in PAIRED:
        s, c, npos, nneg = _score_paired(cls, tgt)
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    for tgt, (s, c, npos, nneg) in _score_oa(cls).items():
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    for tgt in ('Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'):
        if tgt == 'Contusion':
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt), context_penalty=DEGENERATIVE_MARROW, context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt))
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    if FEATURES['synovitis_backoff'] and out['Synovitis__npos'] == 0 and (out['Synovitis__nneg'] == 0):
        proxy = sum((1 for c in cls if SYNOVIAL_PROXY.search(c) and _polarity(c) == 'positive'))
        eff = out['Effusion']
        prior = 0.3 + 0.3 * max(0.0, (eff - 0.5) / 0.45) + 0.06 * min(proxy, 3)
        out['Synovitis'] = min(0.72, prior)
        out['Synovitis__conf'] = 0.18
    return out


# %%%% CELL 3 [code] len=56045
# 2) 数据 / 模型 / 训练 / 推理
"""
RSNA Knee Abnormality Detection —— 统一训练/推理 pipeline

设计目标（Kaggle 约束）：
  - 单次 T4x2 会话（<=2h，硬上限 9h）内跑完 解码 -> 训练 -> 测试推理 -> 写 submission.csv
  - 全程不依赖别人在比赛数据上训出的权重；只用公开预训练（RadImageNet / ImageNet / DINOv2）
  - 不使用 test / LB 做任何反馈（唯一来源是 train.csv + 它的 Report）

数据布局（比赛内置）：
  /kaggle/input/rsna-knee-abnormality-detection/
      train.csv  train_series.csv  test.csv  test_series.csv  sample_submission.csv
      train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
      test_series/<StudyInstanceUID>/...
"""
import os, sys, re, io, json, time, math, glob, random, threading, warnings, multiprocessing as _mp
from dataclasses import dataclass, field
import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# 每列损失权重（由 build_labels 按报告提取器的置信度填），默认全 1
LAB_W = np.ones(len(LAB), np.float32)

# 6 个 slot: (plane, fluid_sensitive)
SLOTS = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1),
         ("Sagittal", 0), ("Coronal", 0), ("Axial", 0)]
SLOT_IDX = {s: i for i, s in enumerate(SLOTS)}


@dataclass
class CFG:
    # ---- 解码 / 采样 ----
    root: str = "/kaggle/input/rsna-knee-abnormality-detection"
    splits_dir: str = "train_series"
    n_slot: int = 5                 # 每个 study 最多用几个 slot（0 位跳过）
    slices_per_slot: int = 12       # 每个 slot 采多少张切片
    img_size: int = 224
    center_frac: float = 0.0        # 0=全序列均匀采样, >0 时偏向中心
    # ---- 训练 ----
    epochs: int = 3
    batch_study: int = 16           # batch 里的 study 数（切片会展开）
    acc_steps: int = 1              # 梯度累积步数：每 acc_steps 个 mini-batch 才 opt.step
                                    # 一次（配合 forward 分块过骨干，把显存峰值从整批压到
                                    # batch_study*acc_steps 张切片内；等效有效 batch 不变）
    lr: float = 3e-4
    weight_decay: float = 1e-4
    backbone: str = "radimagenet"   # radimagenet | imagenet_res50 | dinov2
    emb_dim: int = 2048
    head_hidden: int = 512
    n_val: int = 600                # 内部留出验证的 study 数
    max_train: int = 0              # 0=不限；CPU 冒烟时限制训练 study 数
    time_budget: float = 0.0        # 秒；>0 时超时就跳过训练直接进推理，保证出提交
    n_test_max: int = 0             # 0=全部；冒烟时只推前 N 个 test study
    log_every: int = 20             # 训练/推理每隔多少个 step 打一条心跳（kernel 日志只收 stderr）
    read_smoke: int = 0             # >0 时先读这么几个 study 并逐步打 trace
    seed: int = 0
    amp: bool = False               # 见 discussion 744230：自动混合精度（dtype 按 GPU 选，见 amp_dtype_for）
    # ---- 推理守卫（code-comp 提交重跑时隐藏 test ≈ 1322 study，见 discussion 744230）----
    inf_budget: float = 0.0         # 秒；>0 时推理超时就停并用已完成预测的中位填充，避免常数 0.5
    hidden_test_est: int = 1322     # 外推用：隐藏 test study 数（社区实测 1322）
    # ---- 本地缓存（规避 gcsfuse FUSE 多线程 EBADF）----
    cache_root: str = "/kaggle/working/dcm_cache"  # FUSE 文件 cp 到此（ext4）再读
    precache: bool = False          # True 时启动前整树 cp 到缓存（大批量用）
    cp_study_dir: bool = False      # True 时整 study 目录 cp（v53 及以前的慢路径，默认关）
    # ---- 权重 / 模型 ----
    weights: str = ""               # 权重路径（推理分支用）
    # ---- 输出 ----
    # 关键：Kaggle 提交重跑时从 /kaggle/working/ 提取 submission.csv 打分。
    # 写进 /kaggle/output/ 不会被提交重跑识别（"does not output this file"）。
    # 所以主路径 = /kaggle/working/submission.csv；同时双写 /kaggle/output/ 兜底可下载。
    out: str = "/kaggle/working/submission.csv"


def amp_dtype_for(dev):
    """按 GPU 算选 AMP dtype——别盲抄 bf16。

    T4 / P100 没有 bf16 单元，PyTorch 不报错而是**软件模拟**，前向慢 ~50 倍；
    提交重跑 1322 个隐藏 study 时会大量撞上时间守卫、剩下全填常数 0.5，
    0.851 -> 0.710 就是这么丢的（discussion 744230）。
    返回 (dtype, enabled)。Ampere 及以上（sm80+）才用 bf16。
    """
    if not str(dev).startswith("cuda"):
        return torch.float32, False
    try:
        cc = torch.cuda.get_device_capability(dev)
    except Exception:
        return torch.float16, True
    if cc >= (8, 0):
        return torch.bfloat16, True
    return torch.float16, True


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def log(*a):
    # Kaggle 的 kernel 日志只回收 stderr，stdout 会丢 —— 所以统一走 stderr
    import sys
    print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


def _root_ok(p):
    """一个目录要像数据根目录，至少得有 train_series / test_series 之一，
    或者 train.csv + sample_submission.csv。"""
    for d in ("train_series", "test_series"):
        if os.path.isdir(os.path.join(p, d)):
            return True
    return (os.path.isfile(os.path.join(p, "train.csv"))
            and os.path.isfile(os.path.join(p, "sample_submission.csv")))


def find_root(cfg: CFG, max_depth: int = 5):
    """定位数据根目录。

    比赛数据挂载位置在不同 kernel 类型下并不一致（见过
    /kaggle/input/<dataset> 与 /kaggle/input/competitions/<slug>/<dataset>
    两种），所以这里做有界 BFS，别再假设只有一层。
    """
    cands = [cfg.root, "/kaggle/input", "/kaggle/working", "/kaggle", "."]
    seen = set()
    for c in cands:
        if not os.path.isdir(c) or c in seen:
            continue
        seen.add(c)
        if _root_ok(c):
            return c
        stack = [(c, 0)]
        while stack:
            cur, d = stack.pop()
            if d >= max_depth:
                continue
            try:
                kids = sorted(os.listdir(cur))
            except Exception:
                continue
            for sub in kids:
                p = os.path.join(cur, sub)
                if not os.path.isdir(p) or p in seen:
                    continue
                if _root_ok(p):
                    return p
                stack.append((p, d + 1))
    # 诊断：数据挂载路径没猜中，把现场打印进 stderr（kernel 日志唯一能看到的通道）
    log("!! find_root 失败，现场诊断：")
    log("   cwd =", os.getcwd())
    for c in ["/kaggle", "/kaggle/input", "/kaggle/working", "."]:
        if os.path.isdir(c):
            try:
                log(f"   {c} =", sorted(os.listdir(c)))
            except Exception as e:
                log(f"   {c} 读取失败", e)
        else:
            log(f"   {c} 不存在")
    raise RuntimeError("找不到数据根目录")


# --------------------------------------------------------------------------- #
# DICOM 读取
# --------------------------------------------------------------------------- #
try:
    import pydicom
except Exception:                                    # pragma: no cover
    pydicom = None

# --------------------------------------------------------------------------- #
# 最小 DICOM 解析器
# --------------------------------------------------------------------------- #
# 比赛禁止联网 -> notebook 里 !pip install 不可用，而 pydicom 未必预装。
# 这里自带一个够用的显式/隐式 VR LE 解析器：只要 Rows/Columns/PixelData 就够，
# 强度重标定对分位窗无影响，直接跳过。速度还比 pydicom 快一截。
# 显式 VR LE：这两类 VR 用 2 字节长度，其余（OB/OW/OF/SQ/UT/UN/UC/UR）用 4 字节
_VR2 = {"US", "SS", "UL", "SL", "FL", "FD", "IS", "DS", "LO", "SH", "ST", "UT",
        "CS", "PN", "TM", "DA", "DT", "UI", "AE", "AS", "AT", "LT", "UC", "UR"}
_VR4 = {"OB", "OW", "OF", "SQ", "UN", "OD", "OL", "OV"}
# 我们真正关心的 tag（其余一律跳过长度，反正要跳过）
_KEEP = {
    (0x0028, 0x0002), (0x0028, 0x0010), (0x0028, 0x0011), (0x0028, 0x0100),
    (0x0028, 0x0101), (0x0028, 0x0102), (0x0028, 0x1052), (0x0028, 0x1053),
    (0x0020, 0x1041), (0x0020, 0x0032), (0x0020, 0x0037), (0x0018, 0x0050),
}


# 挂载健康度：累计“卡死”的读取线程数到上限就判定挂载已死，后续读取直接跳过，
# 保证 kernel 一定能“完成”（写出弱提交）而不会被堆积的卡死线程拖到 OOM/段错误。
_WEDGED = {"n": 0}
_WEDGED_LIMIT = 3


def _read_bytes(path, timeout=10.0, tries=2):
    """在守护线程里把整个文件读进内存，带超时（兜底“卡死”的本地 read）。

    先经 _local_path 把 FUSE 文件拷到本地 ext4（单线程 cp），再在线程里读本地
    文件（ext4 线程安全，不会 EBADF）。主进程/线程绝不直读 FUSE。
    """
    path = _local_path(path)
    if _WEDGED["n"] >= _WEDGED_LIMIT:
        return None
    for _ in range(tries):
        box = {}
        def _go():
            try:
                with open(path, "rb") as f:
                    box["d"] = f.read()
            except Exception:
                box["d"] = None
        th = threading.Thread(target=_go, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():               # 读卡死（挂载挂死）——计一次，放弃本次
            _WEDGED["n"] += 1
            if _WEDGED["n"] >= _WEDGED_LIMIT:
                log("!! 挂载疑似已死（%d 次读卡死），后续读取直接跳过，用零图兜底" % _WEDGED["n"])
            time.sleep(0.1)
            continue
        d = box.get("d")
        if d:
            return d
        time.sleep(0.2)
    return None


def read_dcm(path):
    """返回 dict(tag -> bytes)，或 None（读不到 / 隐式 VR / 异常文件）。

    文件字节只读一次（带超时+重试），挂死的挂载会被 _read_bytes 兜住，
    绝不会让 kernel 卡死。
    """
    data = _read_bytes(path)
    if not data:
        return None
    out = _parse_explicit(data)
    return out or None


def _parse_explicit(data):
    """解析显式 VR LE；遇到未知 VR 时自动降级为隐式 VR LE（跳过 VR 直接读 4 字节长度）。"""
    if not data:
        return {}
    out = {}
    n = len(data)
    i = 132 if data[128:132] == b"DICM" else 128   # preamble 128B + 'DICM' 魔数
    implicit = False
    while i + 8 <= n:
        g = int.from_bytes(data[i:i + 2], "little")
        e = int.from_bytes(data[i + 2:i + 4], "little")
        if g == 0 and e == 0:
            break
        i += 4
        if g == 0xFFFE:                       # item / delimitation
            i += 8
            continue
        if not implicit:
            try:
                vr = data[i:i + 2].decode("ascii")
            except Exception:
                break
            if vr in _VR4:
                if i + 6 > n:
                    break
                ln = int.from_bytes(data[i + 2:i + 6], "little"); i += 6
            elif vr in _VR2:
                ln = int.from_bytes(data[i:i + 2], "little"); i += 2
            else:
                # 不是已知显式 VR -> 切到隐式 VR：当前 2 字节当成 4 字节长度的高 2 字节
                implicit = True
                ln = int.from_bytes(data[i:i + 4], "little"); i += 4
        else:
            ln = int.from_bytes(data[i:i + 4], "little"); i += 4
        if ln > n - i:                        # 长度越界，说明 VR 判断错了
            break
        val = data[i:i + ln]
        i += ln
        if (g, e) in _KEEP or (g, e) == (0x7FE0, 0x0010):
            out[(g, e)] = val
        if (g, e) == (0x7FE0, 0x0010):
            break
    return out


def dcm_pixel(d, rows=None, cols=None):
    """从 read_dcm 的结果里还原 2D 像素数组（int16 / uint16 / int32 都支持）。"""
    px = d.get((0x7FE0, 0x0010))
    if px is None:
        return None
    r = d.get((0x0028, 0x0010)) or rows
    c = d.get((0x0028, 0x0011)) or cols
    if r is None or c is None:
        return None
    r = int.from_bytes(r, "little"); c = int.from_bytes(c, "little")
    bits = int.from_bytes(d.get((0x0028, 0x0100), b"\x10\x00"), "little")
    sp = int.from_bytes(d.get((0x0028, 0x0002), b"\x01\x00"), "little")
    if sp != 1:
        return None
    n = r * c
    if bits == 8:
        a = np.frombuffer(px[:n], dtype=np.uint8)
        if int.from_bytes(d.get((0x0028, 0x0101), b"\x08\x00"), "little") == 8:
            a = a.astype(np.int16)
        return a.reshape(r, c)
    if bits == 16:
        a = np.frombuffer(px[: n * 2], dtype="<i2")
        return a.reshape(r, c)
    if bits == 32:
        return np.frombuffer(px[: n * 4], dtype="<i4").reshape(r, c)
    return None

_SLOT_DESC = {
    ("Sagittal", 1): ("sag",), ("Coronal", 1): ("cor",), ("Axial", 1): ("ax",),
    ("Sagittal", 0): ("sag",), ("Coronal", 0): ("cor",), ("Axial", 0): ("ax",),
}


READ_TRACE = False   # 排障时打开：逐个 slot / 逐张切片打日志（kernel 日志只收 stderr）

_READ_DIAG_FIRST = [True]   # 像素读取首个失败原因只打印一次（避免日志刷屏）
# 懒拷贝缓存会把磁盘吃掉（v52：series 目录只拷不删，累积到 ENOSPC -> 训练全崩）。
# 所以这里显式跟踪已拷的文件 + 淘汰（v54 改成逐文件拷贝后，淘汰单位从"目录"变"文件"）。
_CACHE_DIRS = []            # 已拷贝的文件路径列表（FIFO 顺序）
_CACHE_CAP = 6000           # 最多保留 6000 个 dcm 文件（60/study -> 约 100 个 study 的量）
_CACHE_FAIL_N = [0]         # cp 失败计数，每 20 次打一条（别只打第一条，会掩盖规模）
_CACHE_DIRS_CAPPED = [False]


def _cache_note(dst_file):
    """记录一个已拷贝的文件，超上限就删最旧的（防止磁盘写满）。"""
    global _CACHE_DIRS
    _CACHE_DIRS.append(dst_file)
    if len(_CACHE_DIRS) <= _CACHE_CAP:
        return
    _CACHE_DIRS_CAPPED[0] = True
    while len(_CACHE_DIRS) > _CACHE_CAP:
        victim = _CACHE_DIRS.pop(0)
        try:
            os.remove(victim)
        except Exception:
            pass


def _cache_free_gb(path, min_free_gb=2.0):
    """返回 (剩余 GB)；低于 min_free_gb 时返回 False——别再往磁盘拷了。"""
    try:
        import shutil as _sh
        _f = _sh.disk_usage(os.path.abspath(path)).free / (1024.0 ** 3)
        if _f < min_free_gb:
            return False
        return _f
    except Exception:
        return None


def _desc(dcm):
    try:
        return (getattr(dcm, "SeriesDescription", "") or "") + " " + \
               " ".join(getattr(dcm, "ImageType", ()) or ())
    except Exception:
        return ""


def heading_key(dcm):
    """从 DICOM 头猜 (plane, fluid_sensitive)。失败返回 None。"""
    d = _desc(dcm).lower()
    if not d:
        return None
    if any(k in d for k in ("sagittal", "sagitt", "_sag", "sag ")):
        plane = "Sagittal"
    elif any(k in d for k in ("coronal", "coron", "_cor")):
        plane = "Coronal"
    elif any(k in d for k in ("axial", "axiale", "transvers", "_ax")):
        plane = "Axial"
    else:
        return None
    fat = any(k in d for k in ("fat sat", "fatsat", "fs", "stir", "tir", "dixon",
                               "spatial fat", "fsup", "suppress"))
    fs = 1 if fat else 0
    return (plane, fs)


def slice_order_key(dcm):
    """切片排序键：优先 SliceLocation，其次 ImagePositionPatient 的中轴，最后文件名。"""
    try:
        sl = dcm.SliceLocation
        if sl is not None:
            return (0, float(sl), 0.0, "")
    except Exception:
        pass
    try:
        ipp = dcm.ImagePositionPatient
        if ipp:
            # 轴长最大的那一维通常是 KP（sag/cor 沿前后），取其中点做代理键
            axis = int(np.argmax(np.abs(np.array(ipp, dtype=float))))
            return (1, 0.0, float(ipp[axis]), "")
    except Exception:
        pass
    return (2, 0.0, 0.0, "")


_MP_CTX = None

def _mp_ctx():
    global _MP_CTX
    if _MP_CTX is None:
        try:
            _MP_CTX = _mp.get_context("fork")
        except Exception:
            _MP_CTX = _mp.get_context("spawn")
    return _MP_CTX


# --------------------------------------------------------------------------- #
#  本地缓存读取：主进程绝不直读 gcsfuse FUSE
# --------------------------------------------------------------------------- #
#  v39 坐实根因：gcsfuse FUSE 挂载在**多线程进程**里对主线程 read() 也返回
#  EBADF(9)，只有**原生单线程程序**（cp / fork 子进程）能稳定读
#  （PROBE-MAIN 在 BLAS 线程懒加载前成功；PROBE-FORK 子进程稳定 525882B；
#   训练阶段主线程读即 EBADF）。
#  因此统一策略：所有 FUSE 读取都通过单线程 `cp` 子进程拷到本地 ext4，
#  主进程只从本地 ext4 读（线程安全、无 EBADF）。
_LOCAL_CACHE_ROOT = [None]     # 本地缓存根目录（由 main 设置）
_PRECACHE_DONE = [False]       # True 表示整棵输入已 cp 到缓存，直接映射路径


def set_cache_root(root):
    """设置本地缓存根目录（如 /kaggle/working/dcm_cache）。"""
    global _LOCAL_CACHE_ROOT
    _LOCAL_CACHE_ROOT[0] = root
    try:
        os.makedirs(root, exist_ok=True)
    except Exception:
        pass


def mark_precache_done():
    _PRECACHE_DONE[0] = True


def precache_input(data_root, cache_root):
    """（可选）启动前把整棵输入树 cp 到本地缓存——单线程 cp 进程，FUSE 安全。
    返回 True 表示成功。真实大批量运行时用，避免逐文件 cp 的进程开销。"""
    import subprocess as _sp
    dst = os.path.join(cache_root, "input")
    if os.path.isdir(dst) and len(os.listdir(dst)) > 0:
        mark_precache_done()
        return True
    os.makedirs(dst, exist_ok=True)
    try:
        _sp.run(["cp", "-r", "-f", data_root + "/", dst + "/"],
                check=True, timeout=1800,
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        mark_precache_done()
        return True
    except Exception as _e:
        print("PRECACHE fail %s" % repr(_e)[:160], file=sys.stderr, flush=True)
        return False


def _local_path(src):
    """把 FUSE 输入路径映射到本地缓存路径；必要时用单线程 cp 子进程拷过来。
    主进程永不直读 FUSE。"""
    import subprocess as _sp
    root = _LOCAL_CACHE_ROOT[0]
    if root is None or "/kaggle/input" not in src:
        return src
    if _PRECACHE_DONE[0]:
        rel = src.split("/kaggle/input/", 1)[1]
        return os.path.join(root, "input", rel)
    # v54：**逐文件**拷贝（v53 及以前是"整 series 目录"一次 cp）。
    # 一个 series 目录往往有 50~200 张 dcm，而我们每 slot 只要 12 张——整目录 cp
    # 把 v53 训练集 250 个 study 的 cp 总量推到 100GB+，训练 13382s 里 ~11000s 全在 cp。
    # 改成只 cp 真正要读的那一个文件：数据量降到 1/10，cp 次数 90000 次 x ~5ms ≈ 450s。
    rel = src.split("/kaggle/input/", 1)[1]
    dst_file = os.path.join(root, "input", rel)
    if os.path.exists(dst_file) and os.path.getsize(dst_file) > 0:
        return dst_file
    # 磁盘守卫：剩余空间 < 0.5GB 就别再拷了（v52 就是因为只拷不删把磁盘写满，
    # 之后所有 cp 失败、读全回退 FUSE，最后 papermill 写 notebook 时 ENOSPC 直接 kernel ERROR）
    _fg = _cache_free_gb(os.path.dirname(dst_file) or root, min_free_gb=0.5)
    if _fg is False:
        _CACHE_FAIL_N[0] += 1
        if _CACHE_FAIL_N[0] % 20 == 1:
            print("LOCALCP skip(no disk, cached=%d, cached_capped=%s) %s"
                  % (len(_CACHE_DIRS), _CACHE_DIRS_CAPPED[0], src[:120]),
                  file=sys.stderr, flush=True)
        return src
    try:
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        _sp.run(["cp", "-f", src, dst_file], check=True, timeout=60,
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    except Exception as _e:
        _CACHE_FAIL_N[0] += 1
        if _CACHE_FAIL_N[0] % 20 == 1:
            print("LOCALCP fail(cached=%d, fail#%d) %s %s"
                  % (len(_CACHE_DIRS), _CACHE_FAIL_N[0], src[:120], repr(_e)[:120]),
                  file=sys.stderr, flush=True)
        return src
    _cache_note(dst_file)
    return dst_file if os.path.exists(dst_file) else src


def _copy_tree(src, dst):
    """单线程 cp 把整个目录（含子目录）拷到本地 ext4，返回 dst 或 None。
    用于按 study 整目录拷贝——一次覆盖该 study 所有 series 的所有文件读取，
    后续 os.listdir / 头读取 / 像素读取全走本地，消除 FUSE 目录列举与逐文件 cp 开销。
    主进程绝不直接读 FUSE。"""
    import subprocess as _sp, shutil as _sh
    try:
        _sh.rmtree(dst, ignore_errors=True)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _sp.run(["cp", "-r", "-f", src + "/", dst + "/"],
                check=True, timeout=300,
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        return dst
    except Exception as _e:
        if _READ_DIAG_FIRST[0]:
            _READ_DIAG_FIRST[0] = False
            print("COPYTREE fail %s %s" % (src, repr(_e)[:160]),
                  file=sys.stderr, flush=True)
        return None


def read_pixels(path, timeout=30.0):
    """读一个 DICOM 的像素，返回 2D float32 或 None。

    先经 _local_path 把 FUSE 文件拷到本地 ext4（单线程 cp），再在本地读，
    主进程不直接碰 FUSE，彻底规避 EBADF。
    """
    src = _local_path(path)
    try:
        with open(src, "rb") as f:
            data = f.read()
    except Exception as _e:
        if _READ_DIAG_FIRST[0]:
            _READ_DIAG_FIRST[0] = False
            print("READPIX open fail %s %s" % (src, repr(_e)[:160]),
                  file=sys.stderr, flush=True)
        return None
    if not data:
        return None
    d = _parse_explicit(data)
    if d:
        px = dcm_pixel(d)
        if px is not None:
            return px.astype(np.float32)
    if pydicom is not None:
        try:
            ds = pydicom.dcmread(io.BytesIO(data))
            return np.asarray(ds.pixel_array, dtype=np.float32)
        except Exception as _e:
            if _READ_DIAG_FIRST[0]:
                _READ_DIAG_FIRST[0] = False
                print("READPIX pydicom fail %s %s"
                      % (src, repr(_e)[:160]), file=sys.stderr, flush=True)
    return None


def window_slice(arr, lo=0.5, hi=99.5):
    """按分位窗把一片压到 0~255 的 uint8。MRI 强度跨设备不可比，只能用分位窗。"""
    a = arr.astype(np.float32, copy=False)
    # 注意：不能写 a = a[np.isfinite(a)] —— 布尔索引会把 2D 拍成 1D
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == 0:
        return np.zeros(arr.shape, np.uint8)
    p1, p2 = np.percentile(a, lo), np.percentile(a, hi)
    if p2 - p1 < 1e-3:
        p1, p2 = float(a.min()), float(a.max() or 1.0)
        if p2 - p1 < 1e-3:
            return np.zeros(arr.shape, np.uint8)
    out = (a - p1) / (p2 - p1)
    np.clip(out, 0, 1, out=out)
    return (out * 255).astype(np.uint8)


def resize_u8(img, size):
    from PIL import Image
    if img.shape[0] == size and img.shape[1] == size:
        return img
    im = Image.fromarray(img, mode="L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


class SeriesReader:
    """按 slot 组织一个 study 的切片。缓存 series 目录扫描结果。"""

    def __init__(self, root, splits_dir="train_series", planes_meta=None, n_slot=5,
                 slices_per_slot=12, img_size=224, cp_study=False):
        self.root = os.path.join(root, splits_dir)
        self.n_slot, self.k, self.size = n_slot, slices_per_slot, img_size
        self.cp_study = cp_study     # True 时整 study 目录 cp（默认 False：v54 起逐文件 cp 更快）
        self._scan = {}
        self._pix = {}      # 像素缓存：(study, slots) -> (arr, counts)，读一次多 epoch 复用
        self._hdr = {}      # 切片排序键缓存：(series_dir, fname) -> key，省掉每 epoch 重读 300 个头
        self._meta = planes_meta or {}   # SeriesInstanceUID -> (plane, fs)

    def _key(self, study):
        return os.path.join(self.root, study)

    def _scan_study(self, study):
        """扫描一个 study，返回 {slot: [series 目录, ...]}。

        只存 series 目录、不存每个 .dcm 的完整路径：4407 个 study 全量缓存路径
        会堆到百万级字符串，把 kernel 内存顶爆（表现为读文件时 ENOTCONN / 直接
        kernel died）。读像素时再按需 glob。
        """
        if (self.root, study) in self._scan:
            return self._scan[(self.root, study)]
        base = self._key(study)
        if not os.path.isdir(base):
            self._scan[(self.root, study)] = {}
            return self._scan[(self.root, study)]
        per_slot = {}
        for sid in sorted(os.listdir(base)):
            sd = os.path.join(base, sid)
            if not os.path.isdir(sd):
                continue
            plane, fs = self._meta.get(sid, (None, None))
            if (plane, fs) not in SLOT_IDX:
                # 测试集没有 train_series.csv —— 从 DICOM 头里推
                try:
                    _first = _local_path(os.path.join(sd, sorted(glob.glob(
                        os.path.join(sd, "*.dcm")))[0]))
                    d0 = pydicom.dcmread(_first, stop_before_pixels=True)
                    plane, fs = heading_key(d0)
                except Exception:
                    plane, fs = None, None
                if (plane, fs) not in SLOT_IDX:
                    continue
            per_slot.setdefault((plane, fs), []).append(sd)
        self._scan[(self.root, study)] = per_slot
        return per_slot

    def _files_in(self, dirs):
        """按需取一个 slot 下的 .dcm 路径（只在真正读像素时调用）。"""
        out = []
        for sd in dirs:
            try:
                out.extend(sorted(glob.glob(os.path.join(sd, "*.dcm"))))
            except Exception:
                continue
        return out

    def pick_slot_order(self, per_slot):
        """返回按重要性排序的 slot 列表。"""
        order = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1),
                 ("Sagittal", 0), ("Coronal", 0), ("Axial", 0)]
        got = [s for s in order if s in per_slot]
        return got[: self.n_slot] if self.n_slot > 0 else got

    def _header_key(self, f):
        """只读头，取切片排序键；失败退回文件名。

        v54：结果按 (series_dir, fname) 缓存。一个 study 有 ~5 slot x 60 候选文件，
        排序只用到 self.k 张，但读头要遍历全部候选；多 epoch 下这 300 次读/study
        会重复发生（v53 单 epoch 就 90000 次 DICOM 读）。缓存后降到 60 次/study。
        """
        ck = (os.path.dirname(f), os.path.basename(f))
        v = self._hdr.get(ck)
        if v is not None:
            return v
        try:
            d = read_dcm(f)
            if d is not None and (0x0020, 0x1041) in d:
                v = (0, float(np.frombuffer(d[(0x0020, 0x1041)], "<f4")[0]), 0.0, "")
            elif d is not None and (0x0020, 0x0032) in d:
                vv = np.frombuffer(d[(0x0020, 0x0032)], "<f4")
                ax = int(np.argmax(np.abs(vv))) if len(vv) else 0
                v = (1, 0.0, float(vv[ax]) if len(vv) else 0.0, "")
            else:
                v = None
        except Exception:
            v = None
        if v is None:
            v = (2, 0.0, 0.0, os.path.basename(f))
        if len(self._hdr) < 400000:      # 上限 40 万条，别把 kernel 内存顶爆
            self._hdr[ck] = v
        return v

    def read(self, study, slots=None):
        """公开读接口：先把整个 study 目录单线程 cp 到本地 ext4（规避 gcsfuse
        多线程 EBADF，且免去逐文件 cp / FUSE 目录列举开销），再在本地读，
        读完即删缓存（磁盘峰值只占一个 study）。"""
        # 注意：self.root 在 __init__ 里已经拼过 splits_dir（self.root = root/splits），
        # 不存在 self.splits_dir 属性 —— 这里必须用 _key(study) 拿 study 目录。
        ck = (study, tuple(slots) if slots is not None else None)
        if ck in self._pix:
            return self._pix[ck]          # 命中缓存：跳过整目录 cp + 解码，epoch 复用
        study_dir = self._key(study)
        local = None
        # v54：整 study 目录 cp 默认关掉（逐文件 cp 已经够快，整目录 cp 会把
        # 一个 study 的 10~20 个 series 全拖下来）。开关保留，需要时再开。
        if (self.cp_study and _LOCAL_CACHE_ROOT[0] is not None and "/kaggle/input" in study_dir
                and not _PRECACHE_DONE[0]):
            local = _copy_tree(study_dir,
                               os.path.join(_LOCAL_CACHE_ROOT[0], "run", study))
        saved_root = self.root
        if local is not None:
            self.root = os.path.join(_LOCAL_CACHE_ROOT[0], "run")
        try:
            res = self._read_body(study, slots)
        finally:
            if local is not None:
                self.root = saved_root
                import shutil as _sh
                try:
                    _sh.rmtree(local)
                except Exception:
                    pass
        # 缓存上限 2500 个 study，超过就不再缓存（避免内存爆；读仍正确，只是慢）
        if len(self._pix) < 2500:
            self._pix[ck] = res
        return res

    def _read_body(self, study, slots=None):
        """返回 (arr, counts)：

        arr    : (S, H, W) uint8, S = n_slot * k
        counts : 长度 n_slot 的 list，counts[j] = 第 j 个 slot 真实读到的切片数。
                 （切片按 slot-major 排列，所以必须逐 slot 记，单看总数对不上。）

        读不到返回 (None, [0]*n_slot)。

        两步走：先只读 DICOM 头（stop_before_pixels）做切片排序 + 选层，
        再对挑中的那几张读像素。820k 文件里只动 ~5% 的像素读取。
        """
        per_slot = self._scan_study(study)
        chosen = slots or self.pick_slot_order(per_slot)
        if not chosen:
            return None, [0] * self.n_slot
        chosen_files, per_slot_cnt = [], []
        for _si, slot in enumerate(chosen):
            if READ_TRACE:
                log(f"    read: slot {_si}/{len(chosen)} {slot}")
            # 显式转 list + int() 下标：deque / numpy index 混用会炸
            files = list(per_slot.get(slot) or [])
            if not files:                     # slot 存在但目录为空
                per_slot_cnt.append(0)
                continue
            files = self._files_in(files)     # 现在存的是 series 目录
            if not files:
                per_slot_cnt.append(0)
                continue
            if not files:
                per_slot_cnt.append(0)
                continue
            if len(files) > self.k:
                # 不能 np.argsort([tuple,...])：等长元组会被压成 (n,4) 二维，
                # 再取标量下标每行全是 list。显式按 key 排序后按位置抽样。
                keyed = [(str(self._header_key(f)), i) for i, f in enumerate(files)]
                keyed.sort(key=lambda t: t[0])
                n = len(keyed)
                m = min(self.k, n)
                sel = sorted({int(p) for p in
                              np.linspace(0, n - 1, m, dtype=np.int64).tolist()})
                # keyed[i][1] 是原始下标（int），要映射回原文件路径
                files = [files[keyed[i][1]] for i in sel]
            chosen_files.extend(files)
            per_slot_cnt.append(len(files))
        out = []
        for _fi, f in enumerate(chosen_files):
            if READ_TRACE:
                log(f"    read: pixel {_fi+1}/{len(chosen_files)}")
            px = read_pixels(f)
            if px is None or px.ndim != 2 or px.size == 0:
                continue
            if READ_TRACE:
                log("    read: window+resize")
            out.append(resize_u8(window_slice(px), self.size)[None])
        if not out:
            return None, [0] * self.n_slot
        arr = np.concatenate(out, 0).astype(np.uint8, copy=False)
        need = self.n_slot * self.k
        nvalid = len(arr)
        if nvalid < need:
            pad = np.zeros((need - nvalid, self.size, self.size), np.uint8)
            arr = np.concatenate([arr, pad], 0)
        arr = arr[:need]
        counts = list(per_slot_cnt) + [0] * (self.n_slot - len(per_slot_cnt))
        return arr, counts[:self.n_slot]


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class _TinyEnc(nn.Module):
    """冒烟测试 + 效率分支：3 层小卷积，无预训练权重依赖。"""

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        f = self.stem(x)
        return f.view(f.size(0), -1)


class AttnPool(nn.Module):
    """learned query attention over a stack of slice embeddings。

    注意：padding 位必须在 **softmax 之前** 用 -inf 屏蔽，而且不能让 -inf
    穿过打分器（Linear 里 0*(-inf) 会变成 NaN）。所以 mask 作用在 scores 上，
    不作用在 embedding 上。
    """

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask=None):
        # x: (B, S, D)  mask: (B, S, 1) bool，True = padding
        scores = self.proj(x).squeeze(-1)               # (B, S)
        if mask is not None:
            scores = scores.masked_fill(mask.squeeze(-1), float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, S, 1)
        return torch.sum(w * x, dim=1), w


class KneeModel(nn.Module):
    """slice encoder -> 两维(scatter by slot) attention pool -> 12 头。"""

    def __init__(self, backbone="radimagenet", emb=2048, head_hidden=512,
                 n_slot=5, n_cls=12, slices_per_slot=12):
        super().__init__()
        if backbone == "tiny":            # 冒烟测试 / 效率分支用的极简骨干
            self.enc = _TinyEnc()
            self.backbone_conv = None      # tiny 不走分块路径，直接 self.enc(x)
            self.post = None
            self.emb = self.enc.dim
            self.slot_emb = nn.Embedding(n_slot, self.emb)
            self.pool = AttnPool(self.emb)
            self.head = nn.Sequential(
                nn.LayerNorm(self.emb), nn.Linear(self.emb, head_hidden),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(head_hidden, n_cls))
            self.attn_maps = None
            self.n_slice = n_slot * slices_per_slot
            self.n_slot = n_slot
            return
        import torchvision.models as M
        # 注意：kernel 跑在 enable_internet=False 下，不能下载 ImageNet 权重。
        # 用 weights=None 随机初始化，随后若 RadImageNet 数据集已挂载则覆盖预训练层。
        m = M.resnet50(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.maxpool = nn.Identity()          # 224 -> 28 就够，省算力
        # 剥掉 resnet50 原生的分类头 fc（2048->1000）和最后的 avgpool，
        # 只保留到 layer4，输出 (N,2048,7,7) 交给我们的后处理。
        # 注意：不剥 fc 会在 forward 时把 (N,2048,7,7) 误送进 Linear(2048,1000)
        # -> shape 炸（v50 实测 RuntimeError: mat1 ... and 2048x1000），异常后走兜底写 3 行占位。
        feats = nn.Sequential(*list(m.children())[:-2])
        # 把“卷积骨干”和“后处理（全局池化 + 展平 + 归一）”拆开：
        # forward 里卷积骨干按 64 张一批分块过，避免 960 张图一次性占爆显存
        # （v49 OOM 根因：Tried to allocate 5.74 GiB on a 4.00 GiB-free GPU）；
        # 后处理在整批 (N,2048,7,7) 上做，BatchNorm 仍能看到整批统计量，训练动态不受损。
        self.backbone_conv = feats
        self.post = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.BatchNorm1d(2048))
        self.enc = nn.Sequential(feats, nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.BatchNorm1d(2048))
        d = 2048
        if backbone == "radimagenet":      # 可选：覆盖成 RadImageNet 权重
            p = "/kaggle/input/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt"
            if os.path.exists(p):
                sd = torch.load(p, map_location="cpu")
                sd = sd.get("state_dict", sd)
                cur = self.enc.state_dict()
                new = {}
                for k, v in sd.items():
                    k = k.replace("module.", "")
                    if "layer" in k or "conv" in k or "bn" in k:
                        k = k.replace("fc.", "").replace("classifier.", "")
                        if k.startswith("fc") and not k.startswith("layer"):
                            continue
                    if k in cur and cur[k].shape == v.shape:
                        new[k] = v
                self.enc.load_state_dict(new, strict=False)
                log(f"radimagenet 载入 {len(new)} 层")
        self.emb = d
        self.slot_emb = nn.Embedding(n_slot, d)
        self.pool = AttnPool(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(head_hidden, n_cls),
        )
        self.attn_maps = None
        self.n_slot = n_slot
        self.n_slice = n_slot * slices_per_slot

    def forward(self, x, counts=None):
        """x: (B*S, 1, H, W) 展平后的切片栈；counts: (B, n_slot) 每个 slot 的有效切片数。"""
        B = x.shape[0] // self.n_slice if self.n_slice else x.shape[0]
        S = self.n_slice or x.shape[0]
        if self.backbone_conv is not None:
            # 分块 + 梯度检查点：每块只用 checkpoint 跑卷积骨干，forward 不存中间激活，
            # backward 时重算 → 整批计算图只占“分块输入”那点显存，从 ~14GiB 降到 ~0.1GiB，
            # 彻底规避训练 OOM（v51 实测：batch_study=4 整批 240 张图图把 14.20GiB 占满）。
            # 推理（eval/no_grad）不走 checkpoint，直接跑更省时。
            chunks = []
            for i in range(0, x.shape[0], 64):
                xi = x[i:i + 64]
                if self.training and torch.is_grad_enabled():
                    chunks.append(torch.utils.checkpoint.checkpoint(
                        self.backbone_conv, xi, use_reentrant=False))
                else:
                    chunks.append(self.backbone_conv(xi))
            feat = self.post(torch.cat(chunks, 0))
        else:
            feat = self.enc(x)
        if feat.dim() > 2:
            feat = feat.mean(dim=(2, 3))
        feat = feat.view(B, S, -1)
        # slot 位置编码：切片是 slot-major 排列，第 i 张属于第 (i // k) 个 slot
        nsl = min(self.n_slot, self.slot_emb.num_embeddings)
        k = max(1, S // nsl)
        ids = (torch.arange(S, device=x.device) // k).unsqueeze(0).expand(B, S)
        feat = feat + self.slot_emb(ids)
        mask = torch.zeros(B, S, 1, dtype=torch.bool, device=x.device)
        if counts is not None:
            c = counts.to(x.device)                              # (B, nsl)
            slot_of_s = torch.arange(S, device=x.device) // k     # (S,)
            cj = c.gather(1, slot_of_s.unsqueeze(0).clamp(0, nsl - 1))   # (B,S)
            in_slot = torch.arange(S, device=x.device).unsqueeze(0) % k  # (1,S)
            valid = (in_slot < cj) & (cj > 0)                    # (B,S)
            mask = (~valid).unsqueeze(-1)                        # (B,S,1)
            # 极端情况：整批全被屏蔽时 softmax 会出 NaN，直接放开
            mask = mask & ~(mask.all(dim=1, keepdim=True))
        pooled, w = self.pool(feat, mask)
        self.attn_maps = w
        return self.head(pooled)


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class StudyDS(torch.utils.data.Dataset):
    """返回 (x, counts, y)。counts: 长度 n_slot 的逐 slot 有效切片数。"""

    def __init__(self, studies, reader, labels=None):
        self.studies, self.reader = list(studies), reader
        self.labels = labels

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        a, counts = self.reader.read(s)
        if a is None:
            a = np.zeros((self.reader.n_slot * self.reader.k, self.reader.size,
                          self.reader.size), np.uint8)
            counts = [0] * self.reader.n_slot
        y = self.labels[i] if self.labels is not None else np.zeros(12, np.float32)
        x = torch.from_numpy(a).float().div(255.0).unsqueeze(1)   # (S,1,H,W)
        cv = torch.tensor(counts[:self.reader.n_slot], dtype=torch.long)
        return x, cv, torch.tensor(y, dtype=torch.float32)


def collate(batch):
    xs = torch.cat([b[0] for b in batch], 0)          # (B*S, 1, H, W)
    ns = torch.stack([b[1] for b in batch], 0)
    ys = torch.stack([b[2] for b in batch], 0)
    return xs, ns, ys


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_meta(cfg):
    import pandas as pd
    root = find_root(cfg)
    log("data root:", root)
    tr = pd.read_csv(os.path.join(root, "train.csv")).fillna("")
    ser = pd.read_csv(os.path.join(root, "train_series.csv"))
    meta = {}
    for r in ser.itertuples(index=False):
        meta[getattr(r, "SeriesInstanceUID")] = (r.Anatomical_Plane,
                                                 int(r.Fluid_Sensitive))
    return root, tr, ser, meta


def build_labels(tr, extractor):
    """用报告提取器给每个 study 打软标签；金标 study 用金标。

    注意：列名含空格（'Medial Meniscus'），itertuples 会重命名为 _12 之类，
    所以一律按列名从 DataFrame 取 values，别用 getattr(row, name)。

    顺带把提取器给的 __conf 汇总成每列置信度 -> LAB_W（损失权重）。
    高静默率列（Synovitis 87%、Fracture 80%、Lateral OA 76%）的目标本身
    AUC 只有 0.71~0.84，等权训练等于让噪声主导梯度，必须降权
    （社区 strong pipeline 也是"masks/weights for missing, uncertain targets"）。
    """
    global LAB_W
    cols = {c: tr[c].astype(str).tolist() for c in LAB}
    ids = tr["StudyInstanceUID"].astype(str).tolist()
    reports = tr["Report"].astype(str).tolist()

    gold = {}
    for i, sid in enumerate(ids):
        row = [cols[c][i] for c in LAB]
        if all(v.strip() == "" for v in row):
            continue
        gold[sid] = np.array(
            [1.0 if v.strip() == "1" else 0.0 for v in row], np.float32)

    soft = {}
    conf_sum = np.zeros(len(LAB), np.float64)
    conf_n = 0
    silent = np.zeros(len(LAB), np.float64)
    nfail = 0
    for i, sid in enumerate(ids):
        if sid in gold:
            continue
        try:
            e = extractor(reports[i])
        except Exception:
            nfail += 1
            continue
        soft[sid] = np.array([float(e[c]) for c in LAB], np.float32)
        for j, c in enumerate(LAB):
            conf_sum[j] += float(e.get(c + "__conf", 1.0) or 1.0)
            if float(e.get(c + "__npos", 0) or 0) == 0 and \
                    float(e.get(c + "__nneg", 0) or 0) == 0:
                silent[j] += 1.0
            conf_n += 1
    if nfail:
        log(f"报告提取失败 {nfail} 条（已跳过，这些 study 无标签）")
    if conf_n:
        conf_mean = conf_sum / conf_n
        # 置信度 0.05（纯静默兜底）-> 权重 0.35；0.85 以上 -> 1.0
        LAB_W[:] = np.clip(conf_mean / 0.85, 0.35, 1.0).astype(np.float32)
        log("每列报告目标置信度 -> 损失权重：")
        for c, cm, sv, w in zip(LAB, conf_mean, silent / max(conf_n, 1), LAB_W):
            log("  %-17s conf=%.3f 静默率=%.1f%% w=%.2f" % (c, cm, 100 * sv, w))
    return gold, soft


def label_audit(gold, soft):
    """在 58 个金标 study 上打分标"目标"本身的 AUC（不看影像）。

    这是社区 strong pipeline 的标准体检项：目标质量差，再强的模型也上不去。
    """
    sids = [s for s in sorted(set(list(gold) + list(soft)))]
    best = {}
    for sid in sids:
        y = gold.get(sid)
        p = soft.get(sid)
        if y is None or p is None:
            continue
        y = np.asarray(y, np.float32)
        if y.sum() == 0:
            continue
        # 金标 study 里挑预测概率最高的那一列（=该 study 真正的问题）
        j = int(np.argmax(p))
        best.setdefault(j, []).append(float(y[j]))
    if not best:
        return
    lines = []
    for j in sorted(best):
        v = np.asarray(best[j], np.float64)
        if v.min() == v.max():
            continue
        lines.append("%s=%.3f(n=%d)" % (LAB[j], v.mean(), len(v)))
    if lines:
        log("金标 study 的目标阳性率（有效监督信号强度）: " + " ".join(lines))


def auroc_macro(y_true, y_prob):
    out = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_prob[:, i]
        n1, n0 = int((yt == 1).sum()), int((yt == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        order = np.argsort(yp, kind="mergesort")
        r = np.empty(len(yp), float)
        sv = yp[order]
        j = 0
        while j < len(sv):
            k = j
            while k + 1 < len(sv) and sv[k + 1] == sv[j]:
                k += 1
            r[order[j:k + 1]] = (j + k) / 2.0 + 1.0
            j = k + 1
        out.append((r[yt == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return float(np.mean(out)) if out else float("nan")


def load_extractor():
    """notebook 里 report_extract 已被拼进同一个 kernel，直接找模块即可。"""
    g = globals()
    for name in ("rex", "report_extract", "REPORT_EXTRACT"):
        o = g.get(name)
        if o is not None and hasattr(o, "extract"):
            return o
    # notebook 里 extract 是裸函数（report_extract.py 顶层就叫 extract），
    # 直接返回函数本身 —— 别再套一层 SimpleNamespace，那样不是 callable。
    if callable(g.get("extract")):
        g["rex"] = g["extract"]
        return g["extract"]
    import importlib.util
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_extract.py"),
              "report_extract.py", "/kaggle/working/report_extract.py"]:
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("rex", p)
            rex = importlib.util.module_from_spec(spec); spec.loader.exec_module(rex)
            return rex
    raise RuntimeError("找不到报告提取器 report_extract.py")


def discover_test_studies(root):
    """真实提交时 test.csv / test_series/ 会被替换成约 1300 个 study。

    这里**动态发现**，绝不写死那 3 个示例 UID。
    """
    import pandas as pd
    tp = os.path.join(root, "test.csv")
    if os.path.isfile(tp):
        try:
            ids = [str(x) for x in pd.read_csv(tp).StudyInstanceUID.astype(str).tolist()]
            ids = [i for i in ids if i and i != "nan"]
            if ids:
                return ids
        except Exception as e:
            log("test.csv 读取失败", e)
    dr = os.path.join(root, "test_series")
    if os.path.isdir(dr):
        return sorted(d for d in os.listdir(dr) if os.path.isdir(os.path.join(dr, d)))
    return []


def main(cfg: CFG, extractor=None):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    import pandas as pd
    # 本地缓存根：所有 FUSE 读取先经单线程 cp 拷到本地 ext4 再读（规避 EBADF）
    set_cache_root(cfg.cache_root)
    if cfg.precache:
        log("stage: precache input -> local ext4 (single-thread cp)")
        precache_input(cfg.root, cfg.cache_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device:", dev)

    log("stage: load_meta")
    root, tr, ser, meta = load_meta(cfg)
    if extractor is None:
        extractor = load_extractor()

    log("stage: build_labels")
    gold, soft = build_labels(tr, extractor)
    log(f"stage: build_labels done gold={len(gold)} soft={len(soft)}")
    label_audit(gold, soft)
    y_by_study = {k: (gold.get(k, soft.get(k))) for k in gold}
    y_by_study.update(soft)
    studies = list(tr.StudyInstanceUID)
    log(f"金标 {len(gold)} / 软标签 {len(soft)} / 总计 {len(studies)}")

    reader = SeriesReader(root, splits_dir=cfg.splits_dir, planes_meta=meta,
                          n_slot=cfg.n_slot, slices_per_slot=cfg.slices_per_slot,
                          img_size=cfg.img_size, cp_study=cfg.cp_study_dir)

    # 只扫描真正要用的 study。
    # 曾经先全量扫 4407 个 study（≈2.6 万次目录遍历），把 /kaggle/input 的网络挂载
    # 拖到 open() 报 ENOTCONN/EBADF，最后 kernel 直接崩。改成先切分、再按需扫描。
    log("stage: split train/val")
    rng = random.Random(cfg.seed); rng.shuffle(studies)
    withlab = [s for s in studies if y_by_study.get(s) is not None]
    val_list = withlab[:cfg.n_val]
    pool = withlab[cfg.n_val:]
    if cfg.max_train and len(pool) > cfg.max_train:
        log(f"限制训练 study: {len(pool)} -> {cfg.max_train}")
        pool = pool[:cfg.max_train]
    train_list = []

    def _ok(s):
        try:
            return bool(reader._scan_study(s))
        except Exception:
            return False

    log("stage: scan train studies")
    train_list = [s for s in pool if _ok(s)]
    log(f"训练可读 study: {len(train_list)} / {len(pool)}")
    val_list = [s for s in val_list if _ok(s)]
    log(f"验证可读 study: {len(val_list)} / {len(withlab[:cfg.n_val])}")
    if cfg.max_train and len(train_list) > cfg.max_train:
        log(f"限制训练 study: {len(train_list)} -> {cfg.max_train}")
        train_list = train_list[:cfg.max_train]

    # 读像素冒烟：崩在这里（kernel died）说明是 DICOM 解析/PIL 侧的问题
    if cfg.read_smoke:
        global READ_TRACE
        READ_TRACE = True
        log("stage: read smoke")
        for _s in train_list[:cfg.read_smoke]:
            log("  ->", _s)
            try:
                _a, _c = reader.read(_s)
                log("  <- ok", None if _a is None else (_a.shape, _a.dtype), _c)
            except Exception as _e:
                log("  <- 异常", type(_e).__name__, _e)
        READ_TRACE = False

    model = KneeModel(cfg.backbone, cfg.emb_dim, cfg.head_hidden,
                      n_slot=cfg.n_slot, slices_per_slot=cfg.slices_per_slot).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 等效优化步/epoch = 训练 study 数 // 有效 batch（batch_study * acc_steps）
    _eff_bs = max(1, cfg.batch_study * max(1, int(cfg.acc_steps)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, steps_per_epoch=max(1, len(train_list) // _eff_bs + 1),
        epochs=cfg.epochs, pct_start=0.2)

    tr_ds = StudyDS(train_list, reader, [y_by_study[s] for s in train_list])
    va_ds = StudyDS(val_list, reader, [y_by_study[s] for s in val_list])
    # CPU kernel 上多 worker 容易互相饿死，冒烟档用 0 并靠心跳定位卡点
    # 数据读取是瓶颈（逐 study 单线程 cp + 本地解码），必须并行。
    # fork 是 COW，worker 不重复占 torch 内存；CPU 内核一样能开，只是要留出余量。
    _ncpu = os.cpu_count() or 1
    nw = min(4, _ncpu) if not torch.cuda.is_available() else min(6, _ncpu)
    log(f"DataLoader workers = {nw}")
    tr_dl = torch.utils.data.DataLoader(tr_ds, cfg.batch_study, shuffle=True,
                                        collate_fn=collate, num_workers=nw, drop_last=True)
    va_dl = torch.utils.data.DataLoader(va_ds, cfg.batch_study, shuffle=False,
                                        collate_fn=collate, num_workers=nw)

    amp_dt, amp_on = amp_dtype_for(dev)
    amp_on = bool(amp_on and cfg.amp)
    log("stage: amp dtype=%s enabled=%s" % (amp_dt, amp_on))
    # 梯度累积：每 acc_steps 个 mini-batch 才 opt.step 一次，等效有效 batch =
    # batch_study * acc_steps，但前向/反向峰值只占 batch_study 张切片 + backbone
    # 内部 64 张分块，显存远低于“整批一次过”（v49 OOM 的双保险）。
    acc = max(1, int(cfg.acc_steps))
    eff = cfg.batch_study * acc
    log("stage: train eff_batch=%d (batch_study=%d x acc_steps=%d)"
        % (eff, cfg.batch_study, acc))
    best = -1.0
    t_start = time.time()
    for ep in range(cfg.epochs):
        if cfg.time_budget and time.time() - t_start > cfg.time_budget:
            log("超时，跳过后续 epoch，直接用当前权重推理")
            break
        model.train(); t0 = time.time(); run = []
        opt.zero_grad(set_to_none=True)
        nab = 0
        for step, (xb, nb, yb) in enumerate(tr_dl):
            xb, nb, yb = xb.to(dev), nb.to(dev), yb.to(dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dt, enabled=amp_on):
                p = model(xb, nb)
            # per-label 权重：高静默率列的目标噪声大，等权会把梯度带偏
            loss = F.binary_cross_entropy_with_logits(
                p.float(), yb, weight=torch.as_tensor(LAB_W, device=dev))
            (loss / acc).backward()      # 累积梯度（loss 已按 acc 归一，保证量级不变）
            nab += 1
            if nab % acc == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                run.append(loss.item())
                if cfg.log_every and (nab // acc) % cfg.log_every == 0:
                    log(f"  ep{ep+1} step {nab // acc} loss={loss.item():.4f} "
                        f"({time.time()-t0:.0f}s)")
                # 全局软截止：读是瓶颈时 epoch 边界检查太晚（v52 就是读爆了、
                # 直到 ENOSPC 崩盘都没触发），这里每个 opt step 都看一眼总预算，
                # 超了就丢弃当前 epoch 的尾巴，用现有权重直接进验证/推理。
                if cfg.time_budget and (time.time() - t_start) > cfg.time_budget:
                    log("全局预算到（%.0fs），中断 epoch %d，用当前权重直接推理"
                        % (time.time() - t_start, ep + 1))
                    break
        # 尾巴：drop_last=True 通常无残留；若有则补一次 step，避免丢最后几批梯度
        if nab % acc != 0:
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append(loss.item())
        model.eval(); probs, ys = [], []
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dt,
                                             enabled=amp_on):
            for xb, nb, yb in va_dl:
                probs.append(model(xb.to(dev), nb.to(dev)).float().cpu()); ys.append(yb)
        P = torch.cat(probs).numpy(); Y = torch.cat(ys).numpy()
        a = auroc_macro(Y, P)
        log(f"epoch {ep+1}: loss={np.mean(run):.4f}  val_macroAUC={a:.4f}  "
            f"({time.time()-t0:.0f}s)")
        if a == a and a > best:   # a==a 过滤 nan（小 val 里某些列单类时 roc 无定义）
            best = a
            try:
                os.makedirs("/kaggle/output", exist_ok=True)
                torch.save(model.state_dict(), "/kaggle/output/best.pt")
            except Exception as e:  # 权重只是副产物，别为它炸掉整轮提交
                log("save best.pt 失败", e)

    # ---------------- 推理 ----------------
    tstudies = discover_test_studies(root)
    if not tstudies:
        log("!! 没找到任何 test study —— 说明数据挂载异常，退回 sample_submission 形状")
        ss = pd.read_csv(os.path.join(root, "sample_submission.csv"))
        sub = ss.copy()
    else:
        if cfg.n_test_max and len(tstudies) > cfg.n_test_max:
            log(f"限制推理 study: {len(tstudies)} -> {cfg.n_test_max}")
            tstudies = tstudies[:cfg.n_test_max]
        log("test studies:", len(tstudies))
        rdr = SeriesReader(root, splits_dir="test_series", planes_meta={},
                           n_slot=cfg.n_slot, slices_per_slot=cfg.slices_per_slot,
                           img_size=cfg.img_size)
        out = np.zeros((len(tstudies), 12), np.float32)
        tds = StudyDS(tstudies, rdr)
        tdl = torch.utils.data.DataLoader(tds, cfg.batch_study, shuffle=False,
                                          collate_fn=collate, num_workers=nw)
        i = 0
        tinf = time.time()
        # 提交重跑是 ~1322 个隐藏 study，而我们 push 时只看到 3 个。
        # 先按实测吞吐外推，超预算就停并用“已完成预测的中位”填补——
        # 填常数 0.5 是社区 0.851->0.710 的直接原因（discussion 744230）。
        deadline = (tinf + cfg.inf_budget) if cfg.inf_budget > 0 else None
        hit_guard = [False]
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dt,
                                             enabled=amp_on):
            for step, (xb, nb, _) in enumerate(tdl):
                p = torch.sigmoid(model(xb.to(dev), nb.to(dev)).float()).cpu().numpy()
                out[i:i + len(p)] = p; i += len(p)
                if cfg.log_every and (step + 1) % cfg.log_every == 0:
                    el = time.time() - tinf
                    log(f"  infer {i}/{len(tstudies)} ({el:.0f}s, "
                        f"{el / max(i, 1):.1f} s/study)")
                    if i > 4:
                        est = el / i * cfg.hidden_test_est / 3600.0
                        left = (deadline - time.time()) / 3600.0 if deadline else -1
                        log(f"   外推 {cfg.hidden_test_est} study ≈ {est:.1f} h"
                            + ("" if left < 0 else f" | 剩余预算 {left:.1f} h"))
                if deadline is not None and time.time() > deadline:
                    hit_guard[0] = True
                    log("推理超时守卫触发，停在第 %d/%d study" % (i, len(tstudies)))
                    break
        if hit_guard[0] and i < len(tstudies):
            med = (np.median(out[:i], axis=0) if i
                   else np.full(out.shape[1], 0.5, np.float32))
            out[i:] = med.astype(np.float32)
            log("未完成 %d 个 study → 用已预测样本的中位填充" % (len(tstudies) - i))
        sub = pd.DataFrame(out, columns=LAB)
        sub.insert(0, "StudyInstanceUID", tstudies)
    # 概率夹到 (0,1) 开区间，避免 AUC 里出现 0/1 并列
    for c in LAB:
        sub[c] = sub[c].clip(1e-4, 1 - 1e-4)
    _write_submission_both(sub, cfg.out)
    return sub


def infer_only(cfg: CFG):
    """最终提交分支：加载已训好的权重，只跑推理。runtime 最小（效率奖友好）。"""
    import pandas as pd
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = find_root(cfg)
    log("data root:", root)
    model = KneeModel(cfg.backbone, cfg.emb_dim, cfg.head_hidden,
                      n_slot=cfg.n_slot, slices_per_slot=cfg.slices_per_slot)
    sd = None
    for p in ([cfg.weights] if cfg.weights else []) + [
        "/kaggle/input/rsna-knee-best/best.pt",
        "/kaggle/output/best.pt",
        "best.pt",
    ]:
        if p and os.path.isfile(p):
            sd = torch.load(p, map_location="cpu")
            log("weights:", p)
            break
    if sd is None:
        raise FileNotFoundError(f"找不到权重，试过 {cfg.weights!r} 与默认路径")
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()

    tstudies = discover_test_studies(root)
    if not tstudies:
        raise RuntimeError("没找到 test study")
    log("test studies:", len(tstudies))
    rdr = SeriesReader(root, splits_dir="test_series", planes_meta={},
                       n_slot=cfg.n_slot, slices_per_slot=cfg.slices_per_slot,
                       img_size=cfg.img_size)
    out = np.zeros((len(tstudies), 12), np.float32)
    i = 0
    with torch.no_grad():
        for arr, counts in _iter_study_tensors(rdr, tstudies, cfg.batch_study):
            nb = torch.tensor(counts[:cfg.n_slot], dtype=torch.long)
            x = torch.from_numpy(arr).float().div(255.0).unsqueeze(1).to(dev)
            p = torch.sigmoid(model(x, nb.to(dev)).cpu()).numpy()
            out[i:i + len(p)] = p; i += len(p)
    sub = pd.DataFrame(out, columns=LAB)
    sub.insert(0, "StudyInstanceUID", tstudies)
    for c in LAB:
        sub[c] = sub[c].clip(1e-4, 1 - 1e-4)
    _write_submission_both(sub, cfg.out)
    return sub


def _write_submission_both(sub, primary):
    """双写 submission.csv：主路径（cfg.out，默认 /kaggle/working/）+ /kaggle/output/ 兜底。

    Kaggle 提交重跑从 /kaggle/working/ 提取 submission.csv 打分；写 /kaggle/output/ 的
    不会被识别为可提交 output。所以主写 primary，再尽力写一份到 /kaggle/output/ 便于下载。
    任一路径失败不影响另一路。
    """
    import pandas as pd
    targets = [primary]
    for extra in ("/kaggle/working/submission.csv", "/kaggle/output/submission.csv"):
        if extra not in targets:
            targets.append(extra)
    for t in targets:
        try:
            d = os.path.dirname(t) or "."
            os.makedirs(d, exist_ok=True)
            sub.to_csv(t, index=False)
            log("wrote", t, sub.shape)
        except Exception as e:
            log("write %s failed: %r" % (t, e))
    """按 batch yield ((S,H,W) uint8, counts)，带超时保护。"""
    buf = []
    for s in studies:
        a, counts = reader.read(s)
        if a is None:
            a = np.zeros((reader.n_slot * reader.k, reader.size, reader.size), np.uint8)
            counts = [0] * reader.n_slot
        buf.append((a, counts))
        if len(buf) >= bs or (deadline and time.time() > deadline and buf):
            for item in buf:
                yield item
            buf = []
    for item in buf:
        yield item



# %%%% CELL 4 [code] len=3691
# 3) 执行 + 兜底（任何异常都必须留下 submission.csv）
import time
import pandas as _pd

def _ids_from(root):
    """从给定 root 里找出 test study 列表；拿不到就返回 []"""
    import os as _os
    try:
        tp = _os.path.join(root, 'test.csv')
        if _os.path.isfile(tp):
            v = [str(x) for x in _pd.read_csv(tp).StudyInstanceUID.astype(str).tolist()]
            v = [i for i in v if i and i != 'nan']
            if v:
                return v
        dr = _os.path.join(root, 'test_series')
        if _os.path.isdir(dr):
            return sorted(d for d in _os.listdir(dr)
                         if _os.path.isdir(_os.path.join(dr, d)))
    except Exception as e:
        print('  _ids_from(%s) 失败: %r' % (root, e))
    return []

T0 = time.time()
ok = False
try:
    sub = main(CFG(n_slot=5, slices_per_slot=12, img_size=224, epochs=2, n_val=300,max_train=2000, batch_study=4, acc_steps=4, backbone='radimagenet', seed=0,amp=True, log_every=5, time_budget=9000.0, inf_budget=12000.0, out='/kaggle/working/submission.csv'))
    ok = True
except Exception:
    import traceback; traceback.print_exc()
    # 兜底：宁可交一份 0.5 占位，也不能让 kernel 空手挂掉
    try:
        import os as _fo
        ids = []
        for base in ['/kaggle/input', '/kaggle', '.']:
            if not os.path.isdir(base) or ids:
                continue
            for _sd in sorted(os.listdir(base)):
                p = os.path.join(base, _sd)
                if not os.path.isdir(p) or ids:
                    continue
                if os.path.isfile(os.path.join(p, 'test.csv')) or \
                   os.path.isfile(os.path.join(p, 'sample_submission.csv')):
                    ids = ids or _ids_from(p)
                if not ids and os.path.isdir(os.path.join(p, 'test_series')):
                    ids = _ids_from(p)
            if not ids:
                for _dp, _dn, _fn in os.walk(base):
                    if 'test_series' in _dn:
                        ids = _ids_from(_dp)
                        break
        if not ids:
            ss = _pd.read_csv(os.path.join(base, 'test.csv'))
            ids = [str(x) for x in ss.StudyInstanceUID.astype(str)]
        print('fallback ids:', len(ids))
        sub = _pd.DataFrame({'StudyInstanceUID': ids})
        for c in ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA',
                  'Lateral OA','PF OA','Effusion','Synovitis',"Baker's",
                  'Contusion','Fracture']:
            sub[c] = 0.5
        _out = '/kaggle/working/submission.csv'
        for _t in [_out, '/kaggle/working/submission.csv', '/kaggle/output/submission.csv']:
            try:
                _d = os.path.dirname(_t) or '.'
                os.makedirs(_d, exist_ok=True)
                sub.to_csv(_t, index=False)
            except Exception as _e:
                print('  fallback write', _t, 'failed:', _e)
        print('FALLBACK written', sub.shape)
    except Exception:
        import traceback; traceback.print_exc()
        raise
print('total runtime %.0fs, submission shape %s' % (time.time()-T0, sub.shape))
try:
    import os as _fo
    _sub = globals().get('sub')
    _paths = ['/kaggle/working/submission.csv',
              '/kaggle/output/submission.csv',
              'submission.csv']
    for _p in _paths:
        try:
            _d = _fo.path.dirname(_p) or '.'
            _fo.makedirs(_d, exist_ok=True)
            _sub.to_csv(_p, index=False)
            _sz = _fo.path.getsize(_p)
            print('FORCED-WRITE OK %s size=%d exists=%s' % (_p, _sz, _fo.path.exists(_p)))
        except Exception as _fe:
            print('FORCED-WRITE FAIL %s %r' % (_p, _fe))
except Exception as _fe:
    import traceback; traceback.print_exc()
