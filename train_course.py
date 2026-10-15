"""Course TF-IDF experiments: three-fold development CV, frozen holdout, final refit."""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, importlib.metadata, json, platform, time, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from sentiment import load_data

ROOT = Path(__file__).resolve().parent
CANDIDATES = ['lecturer_tfidf_logistic', 'word_bigrams_logistic', 'word_bigrams_balanced_logistic', 'word_bigrams_balanced_logistic_c4', 'word_bigrams_balanced_svm', 'word_char_balanced_svm', 'word_char_balanced_logistic', 'word_bigrams_mlp']

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_candidate(name, seed=42):
    if name == 'lecturer_tfidf_logistic':
        return Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression(C=1, max_iter=2000, random_state=seed))])
    if name not in CANDIDATES:
        raise ValueError(name)
    features = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000, sublinear_tf=True, stop_words=None, dtype=np.float32)
    if name == 'word_bigrams_mlp':
        features.set_params(max_features=10000)
        classifier = MLPClassifier(hidden_layer_sizes=(64,), alpha=.01, learning_rate_init=.001, batch_size=128, max_iter=100, early_stopping=True, validation_fraction=.15, n_iter_no_change=8, random_state=seed)
    elif 'svm' in name:
        classifier = LinearSVC(C=1, class_weight='balanced', dual='auto', max_iter=5000, random_state=seed)
    else:
        classifier = LogisticRegression(C=4 if name.endswith('c4') else 1, class_weight='balanced' if 'balanced' in name else None, max_iter=2000, random_state=seed)
    if name.startswith('word_char'):
        chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=40000, sublinear_tf=True, dtype=np.float32)
        features = FeatureUnion([('word', features), ('char', chars)])
    return Pipeline([('tfidf', features), ('classifier', classifier)])

def scores(y, p):
    return dict(accuracy=float(accuracy_score(y,p)), macro_f1=float(f1_score(y,p,average='macro',zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y,p)), negative_recall=float(recall_score(y,p,pos_label=0,zero_division=0)), positive_recall=float(recall_score(y,p,pos_label=1,zero_division=0)))

def fit(model, x, y):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x,y)
    return time.perf_counter()-start, sorted(set(str(w.message) for w in caught))

def export_course_submission(raw_test, template_path, predictions, output):
    template=pd.read_csv(template_path,dtype=str,keep_default_na=False)
    texts=[r['reviews'] for r in raw_test]
    if list(template.columns)!=['reviews','predictions'] or template.reviews.tolist()!=texts:
        raise ValueError('Template must have reviews,predictions and exactly match test text and row order.')
    labels=np.asarray(predictions)
    if len(labels)!=len(texts) or not np.isin(labels,[0,1]).all():
        raise ValueError('Need one binary prediction per test review.')
    if Path(output).exists():
        raise ValueError('Output already exists; choose a new path.')
    template['predictions']=labels.astype(int)
    template.to_csv(output,index=False)
    check=pd.read_csv(output,keep_default_na=False)
    assert check.reviews.tolist()==texts
    assert check.predictions.tolist()==labels.astype(int).tolist()

def run(args):
    out=Path(args.output).resolve(); src=Path(args.data_dir).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new/empty output folder.')
    raw=load_data(src/'train.json',True); raw['duplicate_key']=raw.text.str.lower()
    if (raw.groupby('duplicate_key').label.nunique()>1).any(): raise ValueError('Conflicting duplicate labels.')
    data=raw.drop_duplicates('duplicate_key').reset_index(drop=True)
    test=load_data(src/'test.json',False)
    raw_test=json.loads((src/'test.json').read_text(encoding='utf-8-sig'))
    template=pd.read_csv(src/'submission.csv',dtype=str,keep_default_na=False)
    if list(template.columns)!=['reviews','predictions'] or template.reviews.tolist()!=[r['reviews'] for r in raw_test]:
        raise ValueError('Resolve test/template alignment first.')
    out.mkdir(parents=True,exist_ok=True)
    dev_idx, hold_idx=train_test_split(np.arange(len(data)),test_size=.2,stratify=data.label,random_state=args.seed)
    dev, hold=data.iloc[dev_idx],data.iloc[hold_idx]
    split=data[['source_row','text','label']].copy(); split['split']='development'; split.loc[hold_idx,'split']='holdout'; split.to_csv(out/'split.csv',index=False)
    audit=dict(original_train_rows=len(raw), unique_train_rows=len(data), duplicate_rows_removed=len(raw)-len(data), original_label_counts={str(k):int(v) for k,v in raw.label.value_counts().items()}, unique_label_counts={str(k):int(v) for k,v in data.label.value_counts().items()}, development_rows=len(dev), holdout_rows=len(hold), test_rows=len(test), test_rows_with_exact_normalised_training_overlap=int(test.text.str.lower().isin(set(data.duplicate_key)).sum()), test_labels_available=False, submission_columns=list(template.columns), submission_template_matches_test_text_and_order=True)
    dump(out/'data_audit.json',audit); print(json.dumps(audit),flush=True)
    fold_rows=[]; summaries=[]
    folds=list(StratifiedKFold(3,shuffle=True,random_state=args.seed).split(dev.text,dev.label))
    manifest=dev[['source_row']].copy(); manifest['validation_fold']=0
    for fold,(_,vi) in enumerate(folds,1): manifest.iloc[vi,manifest.columns.get_loc('validation_fold')]=fold
    manifest.to_csv(out/'cv_fold_membership.csv',index=False)
    for name in CANDIDATES:
        print('CV candidate '+name,flush=True)
        for fold,(ti,vi) in enumerate(folds,1):
            model=build_candidate(name,args.seed)
            seconds,notes=fit(model,dev.iloc[ti].text,dev.iloc[ti].label)
            row=dict(model=name,fold=fold,**scores(dev.iloc[vi].label,model.predict(dev.iloc[vi].text)),fit_seconds=seconds,warnings=' | '.join(notes))
            fold_rows.append(row); pd.DataFrame(fold_rows).to_csv(out/'cv_folds.csv',index=False)
            print(f"  fold {fold}: macro-F1={row['macro_f1']:.4f}, accuracy={row['accuracy']:.4f}, negative recall={row['negative_recall']:.4f} ({seconds:.1f}s)",flush=True)
        rows=pd.DataFrame([r for r in fold_rows if r['model']==name])
        summaries.append(dict(model=name,**{m+'_mean':float(rows[m].mean()) for m in scores([0,1],[0,1])},macro_f1_std=float(rows.macro_f1.std(ddof=1)),total_fit_seconds=float(rows.fit_seconds.sum())))
        pd.DataFrame(summaries).to_csv(out/'cv_results.csv',index=False)
    comparison=pd.DataFrame(summaries).sort_values(['macro_f1_mean','accuracy_mean'],ascending=False)
    selected=str(comparison.iloc[0].model)
    # Freeze selection before predicting any holdout labels.
    dump(out/'selection.json',dict(selected_model=selected,criterion='3-fold development CV macro-F1, accuracy tie-break',holdout_used_for_selection=False))
    comparison.to_csv(out/'cv_results.csv',index=False)
    print('FROZEN SELECTION: '+selected+'; now evaluate holdout once.',flush=True)
    dummy=DummyClassifier(strategy='most_frequent').fit(dev.text,dev.label)
    hold_results={'majority_class':scores(hold.label,dummy.predict(hold.text))}
    for name in dict.fromkeys(['lecturer_tfidf_logistic','word_bigrams_mlp',selected]):
        model=build_candidate(name,args.seed); seconds,notes=fit(model,dev.text,dev.label)
        p=model.predict(hold.text)
        hold_results[name]=dict(**scores(hold.label,p),fit_seconds=seconds,warnings=notes,confusion_matrix_true_rows_predicted_columns=confusion_matrix(hold.label,p,labels=[0,1]).tolist(),classification_report=classification_report(hold.label,p,labels=[0,1],target_names=['negative','positive'],output_dict=True,zero_division=0))
        evidence=hold[['source_row','text','label']].copy(); evidence['prediction']=p; evidence['correct']=evidence.label==p
        if hasattr(model,'predict_proba'): evidence['probability_positive']=model.predict_proba(hold.text)[:,1]
        else: evidence['decision_score_positive']=model.decision_function(hold.text)
        evidence.to_csv(out/(name+'_holdout_predictions.csv'),index=False)
        joblib.dump(model,out/(name+'_holdout_model.joblib'),compress=3)
        print('Holdout '+name+': '+str(scores(hold.label,p)),flush=True)
        if name==selected:
            fig,ax=plt.subplots(figsize=(6.2,5)); ConfusionMatrixDisplay.from_predictions(hold.label,p,labels=[0,1],display_labels=['Negative','Positive'],cmap='Blues',colorbar=False,ax=ax)
            ax.set_title('Selected TF-IDF model: untouched holdout'); fig.tight_layout(); fig.savefig(out/'holdout_confusion.png',dpi=180); plt.close(fig)
    dump(out/'holdout_metrics.json',hold_results)
    fig,ax=plt.subplots(figsize=(10,5.5)); ordered=comparison.iloc[::-1]
    ax.barh(ordered.model,ordered.macro_f1_mean,xerr=ordered.macro_f1_std,color='#276D94',capsize=3); ax.set_xlim(0,1); ax.set_xlabel('Macro-F1: mean across 3 development folds (error bars: SD)'); ax.set_title('Model selection uses development data only'); fig.tight_layout(); fig.savefig(out/'model_comparison.png',dpi=180); plt.close(fig)
    print(f'Refitting {selected} on {len(data)} unique labelled reviews...',flush=True)
    model=build_candidate(selected,args.seed); seconds,notes=fit(model,data.text,data.label); p=model.predict(test.text)
    artifact=dict(pipeline=model,selected_model=selected,labels={0:'negative',1:'positive'},training_text_hashes=[hashlib.sha256(t.encode()).hexdigest() for t in data.duplicate_key],submission_format=['reviews','predictions'],seed=args.seed)
    joblib.dump(artifact,out/'model.joblib',compress=3)
    export_course_submission(raw_test,src/'submission.csv',p,out/'submission.csv')
    np.testing.assert_array_equal(p,joblib.load(out/'model.joblib')['pipeline'].predict(test.text))
    final=dict(model=selected,refit_rows=len(data),fit_seconds=seconds,warnings=notes,test_prediction_counts={str(k):int(v) for k,v in pd.Series(p).value_counts().items()},submission_rows=len(p),submission_sha256=sha(out/'submission.csv'),submission_roundtrip_and_model_reload_verified=True)
    dump(out/'run.json',dict(seed=args.seed,threads=args.threads,python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib','matplotlib','threadpoolctl']},input_sha256={n:sha(src/n) for n in ['train.json','test.json','submission.csv','tfidf_features_example.ipynb'] if (src/n).is_file()},source_sha256={'train_course.py':sha(__file__),'sentiment.py':sha(ROOT/'sentiment.py')},candidate_parameters={n:{k:repr(v) for k,v in build_candidate(n,args.seed).get_params().items()} for n in CANDIDATES},selection=selected,final_refit=final))
    write_report(out,audit,comparison,selected,hold_results,final)
    print('COMPLETE: '+str(out),flush=True)

def write_report(out,audit,comparison,selected,hold,final):
    result=hold[selected]; baseline=hold['lecturer_tfidf_logistic']
    lines=['IE4483 REAL-DATA TRAINING RESULTS','',f'Selected model: {selected}',f"Untouched holdout accuracy: {result['accuracy']:.2%}",f"Untouched holdout macro-F1: {result['macro_f1']:.4f}",f"Untouched holdout negative-review recall: {result['negative_recall']:.2%}",f"Lecturer-feature baseline holdout accuracy: {baseline['accuracy']:.2%}",f"Lecturer-feature baseline holdout macro-F1: {baseline['macro_f1']:.4f}",'','DATA AUDIT',json.dumps(audit,indent=2),'','METHOD',
    'The lecturer notebook supplies default unigram TF-IDF features, not a classifier.',
    'The baseline adds logistic regression to those default features.',
    'Case/whitespace-normalised duplicate reviews are removed before splitting.',
    'Near-duplicates/related reviews may remain; this is not product-independent evaluation.',
    'Reserve a stratified 20% holdout; use 3-fold stratified CV on the other 80%.',
    'Fit TF-IDF separately inside each CV fold, retaining negation words.',
    'Choose highest mean CV macro-F1 (accuracy tie-break) BEFORE observing holdout results.',
    'Word bigrams capture phrases, sublinear TF reduces repeated-word influence, and',
    'rare-term filtering and vocabulary limits control feature growth. Character n-grams',
    'add spelling/subword information. These are not contextual embeddings.',
    'Class-weighted linear models address the large positive/negative imbalance.',
    'MLP: TF-IDF (up to 10000 features) -> Dense(64, ReLU) -> sigmoid binary output.',
    'MLP uses Adam, log loss, L2 alpha .01, learning rate .001, batch size 128,',
    'and training-fold-only internal early stopping by accuracy, up to 100 epochs.',
    'Internal MLP validation rows may inform the unsupervised vocabulary but never the',
    'external CV/holdout vocabulary. CV variation is across folds, not repeated seeds.',
    'All candidate settings and package versions are recorded in run.json.','',
    'DEVELOPMENT CV COMPARISON',comparison.to_string(index=False),'','FROZEN HOLDOUT RESULTS',json.dumps(hold,indent=2),'','FINAL REFIT',json.dumps(final,indent=2),
    'The final selected model is refit on all unique labelled training reviews.',
    'Holdout metrics describe the pre-refit model, not accuracy on the unlabelled test set.',
    'No test labels or template placeholder predictions were used for learning/selection.',
    'Training/test duplicate overlap is disclosed; all predictions come from the fitted model.',
    'The actual template has reviews,predictions; original text/order are preserved exactly.','',
    'ARTIFACTS',
    'model.joblib: final classifier and TF-IDF features. submission.csv: all test predictions.',
    'cv_results.csv, cv_folds.csv: model selection means and fold evidence.',
    'holdout_metrics.json, *_holdout_predictions.csv: metrics and error-analysis examples.',
    'holdout_confusion.png, model_comparison.png: figures for the report.',
    'split.csv, cv_fold_membership.csv, selection.json: split and frozen-selection evidence.',
    'run.json: settings, versions, warnings and input/source checksums.','',
    'REPRODUCE: python train_course.py --output runs/course-repeat --seed 42',
    'The script refuses non-empty output folders and never changes the supplied files.',
    'The holdout is now inspected; further tuning on it would invalidate untouched evaluation.',
    'Use fresh nested evaluation if you undertake further model selection.']
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default=str(ROOT/'data'/'course'))
    parser.add_argument('--output',default=str(ROOT/'runs'/('course-'+datetime.now().strftime('%Y%m%d-%H%M%S'))))
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    with threadpool_limits(limits=args.threads): run(args)

if __name__=='__main__': main()
