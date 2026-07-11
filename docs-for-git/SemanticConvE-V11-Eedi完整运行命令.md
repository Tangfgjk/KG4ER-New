# SemanticConvE V11锛欵edi 鏁版嵁闆嗗畬鏁磋繍琛屽懡浠?
鏈枃妗ｅ彧鍐?Eedi 鏁版嵁闆嗙殑 V11 杩愯娴佺▼銆傛墍鏈夊懡浠ら粯璁ゅ湪椤圭洰鏍圭洰褰曟墽琛岋細

```powershell
cd "C:\Users\29694\Desktop\鎴戠殑鏂囦欢\闄嗗瓙娆ｅ笀濮怽2025闄嗗瓙娆Code-New"
conda activate kg4er_cuda
```

V11 鐨勭洰鏍囨槸閲嶆柊闂幆鐢熸垚鍓嶇疆鏂囦欢锛屽苟鎶婄粨鏋滀繚瀛樺埌锛?
```text
ER/KG4ER-New/data/Eedi/v11/
```

鍘熷鏁版嵁浠嶄粠鏃х洰褰曡鍙栵紝涓嶇Щ鍔ㄣ€佷笉瑕嗙洊锛?
```text
ER/KG4ER/data/Eedi/
```

## 1. 鐢熸垚 MIRT 杈撳叆

```powershell
python ER\KG4ER-New\v11_pipeline\prepare_mirt_inputs_v11.py `
  --dataset Eedi `
  --force
```

浣滅敤锛?
- 浣跨敤 ER 鍥惧凡瀵归綈鐨?Eedi-sub 瀛︾敓銆侀鐩拰浣滅瓟璁板綍锛?- 鐢熸垚 no-Q MIRT 璁粌鎵€闇€杈撳叆锛?- 閬垮厤鎶婂畬鏁?Eedi 褰撴垚 Eedi-sub 浣跨敤銆?
## 2. 璁粌 no-Q MIRT

```powershell
python ER\KG4ER-New\v11_pipeline\train_mirt_noq_v11.py `
  --dataset Eedi `
  --epoch 70 `
  --batch-size 1024 `
  --lr 0.001 `
  --device cuda `
  --force
```

浣滅敤锛?
- 灏?MIRT 浣滀负鏁欒偛瀛︾壒寰佷及璁″櫒锛?- 杈撳嚭棰樼洰闅惧害銆佸尯鍒嗗害鍜屽鐢熻兘鍔涘弬鏁帮紱
- 鍚庣画 EKTM_mirt 鍜?SemanticConvE 閮藉鐢ㄨ繖缁勬暀鑲插鍙傛暟銆?
## 3. 璁粌淇鐗?pyKT DKT 骞跺鍑?stu2know_seq.json

```powershell
python ER\KG4ER-New\v11_pipeline\train_pykt_dkt_v11.py `
  --dataset Eedi `
  --epochs 30 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --device cuda `
  --force
```

浣滅敤锛?
- 澶嶅埗 pyKT DKT 鍒?V11 宸ヤ綔鐩綍锛?- 鍙湪澶嶅埗鐗?pyKT 涓慨鏀?DKT loss锛屼娇鏍囩鏀逛负鈥滀笅涓€姝ョ煡璇嗙偣鍑虹幇鈥濓紱
- 璁粌鏂扮殑 DKT checkpoint锛?- 瀵煎嚭锛?
```text
ER/KG4ER-New/data/Eedi/v11/stu2know_seq.json
```

濡傛灉璁粌宸插畬鎴愶紝鍙兂浠庡凡鏈?checkpoint 閲嶆柊瀵煎嚭锛?
```powershell
python ER\KG4ER-New\v11_pipeline\train_pykt_dkt_v11.py `
  --dataset Eedi `
  --device cuda `
  --skip-train
```

## 4. 璁粌 EKTM_mirt

```powershell
python ER\KG4ER-New\v11_pipeline\train_ektm_mirt_v11.py `
  --dataset Eedi `
  --epochs 30 `
  --batch-size 16 `
  --device cuda `
  --force
```

浣滅敤锛?
- 浣跨敤瀛︾敓浣滅瓟搴忓垪銆侀鐩枃鏈€佺煡璇嗙偣鏄犲皠鍜?MIRT 鍙傛暟璁粌 EKTM_mirt锛?- 淇濆瓨 best checkpoint锛?- 璁粌缁撴潫鍚庡鍑?`stu2know_mastery.json`銆佷範棰樻枃鏈?embedding 鍜岀煡璇嗙偣鏂囨湰 embedding銆?
濡傛灉璁粌宸插畬鎴愶紝鍙兂浣跨敤 best checkpoint 閲嶆柊瀵煎嚭锛?
```powershell
python ER\KG4ER-New\v11_pipeline\train_ektm_mirt_v11.py `
  --dataset Eedi `
  --device cuda `
  --export-only
```

## 5. 鐢熸垚閬楀繕鐩稿叧鏂囦欢

```powershell
python ER\KG4ER-New\v11_pipeline\generate_forgetting_v11.py `
  --dataset Eedi `
  --theta 10000000 `
  --timestamp-unit auto `
  --force
```

杈撳嚭锛?
```text
stu2know_forget.json
stu2ex_forget.json
```

V11 涓?`stu2ex_forget.json` 鎸夐鐩秹鍙婄煡璇嗙偣鐨勯仐蹇樼巼骞冲潎鍊奸噸鏂拌绠楋紝閬垮厤鏃ф祦绋嬩腑鈥滄眰鍜屽悗瓒呰繃 1鈥濈殑闂銆?
## 6. 瀵煎嚭 SemanticConvE 鐗瑰緛鏂囦欢

```powershell
python ER\KG4ER-New\v11_pipeline\export_mirt_features_v11.py `
  --dataset Eedi `
  --force
```

浣滅敤锛?
- 姹囨€诲鐢熴€侀鐩€佺煡璇嗙偣銆佸叧绯绘墍闇€鐗瑰緛锛?- 浣跨敤 EKTM_mirt/Bi-GRU 瀵煎嚭鐨勬枃鏈?embedding锛?- 涓嶅啀浣跨敤 BGE/SentenceTransformer锛?- 鍐欏叆锛?
```text
ER/KG4ER-New/data/Eedi/v11/semantic_kg_features/
```

## 7. 鏋勫缓 V11 ER 鍥?
榛樿鎺ㄨ崘鍏紡浣跨敤鏇寸洿瑙傜殑 sequence 椤癸細

```text
(1 - cos(Q_j, seq_i))^2
```

杩愯锛?
```powershell
python ER\KG4ER-New\v11_pipeline\build_v11_graph.py `
  --dataset Eedi `
  --sequence-term one_minus_cos_sq `
  --force
```

杈撳嚭锛?
```text
stu2ex_recommend.json
stu2ex_recommend_full_precision.json
triples.txt
test_triples.txt
entities.dict
relations.dict
```

濡傞渶淇濈暀鏃у叕寮忓鐓э細

```powershell
python ER\KG4ER-New\v11_pipeline\build_v11_graph.py `
  --dataset Eedi `
  --sequence-term legacy_cos_sq `
  --force
```

## 8. 鏍￠獙 V11 鍓嶇疆鏂囦欢

```powershell
python ER\KG4ER-New\v11_pipeline\validate_v11_front_files.py `
  --datasets Eedi
```

閲嶇偣妫€鏌ワ細

- 瀛︾敓銆侀鐩€佺煡璇嗙偣鏁伴噺鏄惁涓?ER 鍥句竴鑷达紱
- `stu2know_mastery.json`銆乣stu2know_seq.json` 鏄惁缁村害姝ｇ‘锛?- 閬楀繕鐜囨槸鍚﹀湪鍚堢悊鑼冨洿锛?- 鏂囨湰 embedding 鏄惁鏉ヨ嚜 EKTM_mirt/TopicRNNModel锛?- 鏄惁浠嶆贩鍏?BGE embedding銆?
## 9. 鍓嶇疆鎺ㄨ崘鍒嗘暟棰勮瘎浼?
```powershell
python ER\KG4ER-New\v11_pipeline\score_v11_front_files.py `
  --dataset Eedi
```

浣滅敤锛?
- 鐩存帴鐢?V11 鎵嬪伐鎺ㄨ崘璺濈鍋氫竴娆♀€滃墠缃垎鏁扳€濊瘎浼帮紱
- 璺濈瓒婂皬瓒婃帹鑽愶紝鑴氭湰浼氳嚜鍔ㄨ浆鎴愯瘎浼板櫒闇€瑕佺殑楂樺垎浼樺厛鏍煎紡锛?- 杈撳嚭锛?
```text
ER/KG4ER-New/data/Eedi/v11/front_oracle_eval/
```

杩欎竴姝ョ敤浜庢彁鍓嶅垽鏂墠缃枃浠舵湰韬槸鍚﹀紓甯革紝涓嶇瓑浠蜂簬 SemanticConvE 鏈€缁堢粨鏋溿€?
## 10. 璁粌涓庢祴璇?SemanticConvE 涓绘ā鍨嬪拰娑堣瀺瀹為獙

杩涘叆鏂颁唬鐮佺洰褰曪細

```powershell
cd "C:\Users\29694\Desktop\鎴戠殑鏂囦欢\闄嗗瓙娆ｅ笀濮怽2025闄嗗瓙娆Code-New\ER\KG4ER-New"
```

杩愯 Eedi 浜斾釜闅忔満绉嶅瓙鐨勫畬鏁村疄楠岋細

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v11 `
  --run-id Eedi_v11_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto
```

缁窇鍛戒护锛?
```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v11 `
  --run-id Eedi_v11_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

濡傛灉闇€瑕佷繚鎸佹棫浠ｇ爜 transductive 璁剧疆锛屽嵆璁粌鏃跺姞鍏?`test_triples.txt`锛?
```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset Eedi `
  --data-root data `
  --graph-subdir v11 `
  --run-id Eedi_v11_include_test_5seeds `
  --seeds 2024,2025,2026,2027,2028 `
  --ablations full,id_only,no_pedagogical,no_text_semantic,no_concept_semantic,no_relation_aware,no_type_aware_scoring,no_mastery,no_forgetting,no_seq `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --include-test-triples
```

## 11. 缁熻瀹為獙缁撴灉

鍦?`ER/KG4ER-New` 鐩綍涓嬭繍琛岋細

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v11_5seeds `
  --seeds 2024,2025,2026,2027,2028
```

濡傛灉缁熻 include-test 鐗堟湰锛?
```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset Eedi `
  --run-id Eedi_v11_include_test_5seeds `
  --seeds 2024,2025,2026,2027,2028
```

缁撴灉鐩綍锛?
```text
ER/KG4ER-New/runs/Eedi/<run-id>/summaries/
```

## 12. 鏂扮數鑴戣繍琛岄渶瑕佹嫹璐濅粈涔?
濡傛灉鍦ㄦ柊鐢佃剳鍙窇 Eedi锛屾媺鍙?V11 浠ｇ爜鍚庯紝鎷疯礉杩欎釜鐩綍锛?
```text
ER/KG4ER-New/data/Eedi/v11/
```

鏀惧埌鏂扮數鑴戜粨搴撲腑锛?
```text
KG4ER-New/data/Eedi/v11/
```

鐒跺悗杩涘叆锛?
```powershell
cd KG4ER-New
```

鍗冲彲杩愯绗?10 鑺傚拰绗?11 鑺傚懡浠ゃ€?

