"""Build the final evidence-based LHCO project report from generated results."""
from pathlib import Path
import json, math
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image, KeepTogether)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'pdf'/'LHC_Olympics_Project_Report.pdf'
RESULTS=ROOT/'reports'/'tables'/'results.csv'

def nice(x, n=3):
    try:
        if pd.isna(x): return '—'
        return f'{float(x):.{n}f}'
    except (ValueError,TypeError): return str(x)

def make_charts(df):
    p=ROOT/'reports'/'figures'; p.mkdir(parents=True,exist_ok=True)
    made=[]
    base=df[(df.seed==42)&(df.k_labels==0)&(df.f_train.isin([0,.1,.5,1]))]
    fig,ax=plt.subplots(figsize=(8,4.2))
    for name,g in base.groupby('model'):
        ax.plot(g.f_train,g.roc_auc,marker='o',label=name.replace('_',' '))
    ax.set(xlabel='Contaminated signal fraction in training (%)',ylabel='Held-out 2-prong ROC AUC',ylim=(0,1),title='Performance across the contamination grid')
    ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout()
    path=p/'report_contamination_auc.png'; fig.savefig(path,dpi=180); plt.close(fig); made.append(path)
    h=df[(df.seed==42)&(df.model=='M4_DeepSAD')]
    if len(h):
        pv=h.pivot(index='f_train',columns='k_labels',values='roc_auc')
        fig,ax=plt.subplots(figsize=(6.8,3.8)); im=ax.imshow(pv.values,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(len(pv.columns)),[str(x) for x in pv.columns]); ax.set_yticks(range(len(pv.index)),[f'{x:g}%' for x in pv.index])
        ax.set(xlabel='Labeled signal events (k)',ylabel='Training contamination',title='Deep SAD: held-out 2-prong ROC AUC')
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                if pd.notna(pv.iloc[i,j]): ax.text(j,i,f'{pv.iloc[i,j]:.3f}',ha='center',va='center',color='white',fontsize=9)
        fig.colorbar(im,ax=ax,label='ROC AUC'); fig.tight_layout()
        path=p/'report_deepsad_auc.png'; fig.savefig(path,dpi=180); plt.close(fig); made.append(path)
    rep=ROOT/'reports'/'tables'/'replicate_summary.csv'
    if rep.exists():
        r=pd.read_csv(rep,header=[0,1],index_col=[0,1,2])
        fig,ax=plt.subplots(figsize=(7.5,4.1)); names=[]; vals=[]; errs=[]
        for model in ['M1_Autoencoder','M2_IsolationForest','M3_DeepSVDD','M4_DeepSAD','M5_Supervised','M6_MassAware']:
            key=(model,0.5,100) if model=='M4_DeepSAD' else (model,0.0,1000 if model=='M5_Supervised' else 0)
            if key in r.index and ('roc_auc','mean') in r.columns:
                names.append(model.replace('_',' ')); vals.append(r.loc[key,('roc_auc','mean')]); errs.append(r.loc[key,('roc_auc','std')])
        if names:
            fig,ax=plt.subplots(figsize=(8,4.2)); ax.bar(range(len(names)),vals,yerr=errs,capsize=3,color='#1e7188')
            ax.set_xticks(range(len(names)),names,rotation=25,ha='right'); ax.set_ylabel('Held-out 2-prong ROC AUC'); ax.set_ylim(0,1); ax.set_title('Headline comparisons (mean ± SD across seeds)'); ax.grid(axis='y',alpha=.25); fig.tight_layout()
            path=p/'report_headline_auc.png'; fig.savefig(path,dpi=180); plt.close(fig); made.append(path)
    return made

def build():
    if not RESULTS.exists(): raise SystemExit(f'Missing results: {RESULTS}')
    df=pd.read_csv(RESULTS); df=df.drop_duplicates(['model','f_train','k_labels','seed'],keep='last')
    split=json.loads((ROOT/'data/splits/split_manifest.json').read_text(encoding='utf8'))
    fv=[json.loads(x.read_text(encoding='utf8')) for x in sorted((ROOT/'reports').glob('feature_validation_*.json'))]
    manifest=json.loads((ROOT/'reports/tables/run_manifest.json').read_text(encoding='utf8')) if (ROOT/'reports/tables/run_manifest.json').exists() else []
    completed=sum(x.get('status','').startswith(('completed','already completed')) for x in manifest)
    seeds=sorted(df.seed.unique()); charts=make_charts(df)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Cover',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=25,leading=31,textColor=colors.HexColor('#14283f'),alignment=TA_CENTER,spaceAfter=16))
    styles.add(ParagraphStyle(name='Deck',parent=styles['Normal'],fontSize=12,leading=18,textColor=colors.HexColor('#36516b'),alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='H1x',parent=styles['Heading1'],fontSize=17,leading=21,textColor=colors.HexColor('#14283f'),spaceBefore=12,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='H2x',parent=styles['Heading2'],fontSize=11.5,leading=15,textColor=colors.HexColor('#176b7b'),spaceBefore=9,spaceAfter=4,keepWithNext=True))
    styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=6))
    styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontSize=7.5,leading=10,textColor=colors.HexColor('#4c5b68')))
    styles.add(ParagraphStyle(name='Cellx',parent=styles['BodyText'],fontSize=7.2,leading=9))
    styles.add(ParagraphStyle(name='Callout',parent=styles['BodyText'],fontSize=10,leading=15,textColor=colors.HexColor('#14283f'),backColor=colors.HexColor('#eaf2f5'),borderPadding=8,spaceBefore=5,spaceAfter=9))
    def P(s,style='Bodyx'): return Paragraph(s,styles[style])
    def table(rows,widths,fontsize=7.4):
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#14283f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),fontsize),('LEADING',(0,0),(-1,-1),fontsize+2),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#cbd7df')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf3f6')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return t
    story=[Spacer(1,42*mm),P('Unsupervised and Semi-Supervised<br/>Anomaly Detection','Cover'),P('on the LHC Olympics 2020 R&amp;D Dataset','Deck'),Spacer(1,8*mm),P('Final project report · Evidence-based reproduction and implementation audit','Deck'),Spacer(1,5*mm),P('Mithil Hardik Astik · 3 October 2026','Deck'),Spacer(1,24*mm),P('Simulation-only benchmark. This report describes methods and measured performance on public simulated samples; it makes no discovery claim and does not estimate real-data physics sensitivity.','Callout'),PageBreak()]
    story += [P('Executive summary','H1x'),P(f'This project benchmarks six anomaly-detection methods on the public LHC Olympics 2020 R&amp;D dijet samples. The verified Zenodo v5 release supplies 1.1 million 2-prong events for model development and 100,000 3-prong signal events for topology-transfer evaluation. The implementation produces deterministic event splits, validated features, model checkpoints, evaluation tables, mass diagnostics, and run metadata.'),P(f'The reproduction run contains {len(df)} unique evaluated configurations across seeds {", ".join(map(str,seeds))}; {completed} run records are marked completed in the run manifest. The full contamination/label-budget grid is anchored to seed 42. Headline repeated-seed summaries are reported only if the corresponding replicate table exists and includes those runs.'),P('Main findings','H2x')]
    main=df[(df.seed==42)&(df.f_train==0)&(df.k_labels==0)]
    if len(main):
        best=main.sort_values('roc_auc',ascending=False).iloc[0]
        story.append(P(f'Among zero-contamination, zero-labeled-signal models with seed 42, {escape(str(best.model))} has the highest held-out 2-prong ROC AUC ({best.roc_auc:.3f}). This is a single-seed benchmark result, not a claim of statistical superiority. Its 3-prong ROC AUC is {best.roc_auc_3prong:.3f}.' ,'Callout'))
    story += [P('Interpretation is constrained by one-seed grid results, the limited repeated-seed headline subset, simulation-only data, and the known subjettiness-definition discrepancy documented below. Background mass preservation and the fixed-window null are diagnostics; the reported local significance is not a global discovery significance.'),P('Project status','H2x'),P('Completed: official data acquisition and checksum verification; canonical feature ingestion and low-level cross-checks; deterministic class-stratified splits; train/validation/test separation; six method implementations; contamination and label-budget grids; held-out 3-prong scoring; validation-derived operating thresholds; mass-shaping and fixed-window diagnostics; packaging, license, citation metadata, and this report. Outstanding audit limitations and deviations are listed in Section 10.')]
    story += [P('1. Research question and scope','H1x'),P('The central question is how one-class and few-label anomaly detectors trade off detection ranking, robustness to contaminated training data, transfer to a different signal topology, and preservation of the resonant background mass spectrum. The study is a reproducible methods benchmark on simulated R&amp;D data. It does not make a claim about discoveries, detector data, or expected experimental sensitivity.'),P('The full project requirements are captured in PRD_final.md. The experiment uses the released 2-prong signal for development and holds the distinct 3-prong signal sample for transfer evaluation. The resonant observable mJJ is excluded from the principal detector inputs and included only in an explicit leakage-control model.')]
    story += [P('2. Dataset and provenance','H1x'),P('The four public HDF5 files were obtained from LHC Olympics 2020 R&amp;D dataset, Zenodo record v5 (DOI 10.5281/zenodo.6466204). The raw event files contain particle-level detector representations; released high-level feature files provide the canonical benchmark features. The data manifest records source checksums, verified SHA-256 digests, and file sizes. The raw data are not distributed with this repository.'),table([[P('Sample','Cellx'),P('Available rows','Cellx'),P('Use','Cellx')],[P('QCD background','Cellx'),P('1,000,000','Cellx'),P('Train/validation/test background pool','Cellx')],[P('2-prong signal','Cellx'),P('100,000','Cellx'),P('Development signal, contamination, labeled pool and test signal','Cellx')],[P('3-prong signal','Cellx'),P('100,000','Cellx'),P('Held-out topology transfer test','Cellx')]], [38*mm,30*mm,112*mm]),Spacer(1,4),P('Event splits are deterministic and stratified with seed 42. Invalid feature rows are excluded. The 2-prong background split contains 599,952 train, 199,984 validation, and 199,984 test rows; signal counts are 59,983, 19,994, and 19,996. The separate 3-prong sample contributes 99,975 valid held-out events.'),P('6.1 Reproducibility inputs','H2x'),P('The benchmark uses author-supplied high-level features as the canonical values. This avoids silently substituting a locally implemented subjettiness convention for the release convention. A deterministic 128-event raw-clustering sample is compared against those supplied quantities. The optional --from-raw path reclusters all events and documents its exclusive-kT subjettiness axes; it is not the canonical benchmark route.')]
    story += [P('3. Feature construction and validation','H1x'),P('Principal inputs are leading-jet mass (mJ1), mass difference (dmJ), leading and subleading jet tau21, and leading-jet angular separation (dRJJ). The dijet mass mJJ is retained for mass diagnostics and the M6 leakage-control ablation. It is excluded from M1–M5 model input. Tau21 is tau2/tau1 with invalid/zero denominators identified and removed from the usable event set.'),table([[P('Sample','Cellx'),P('Rows','Cellx'),P('Invalid tau rows','Cellx'),P('mJ1 MAE (GeV)','Cellx'),P('dmJ MAE (GeV)','Cellx'),P('dRJJ MAE','Cellx'),P('mJJ MAE (GeV)','Cellx')]]+[[P(str(x['canonical_source']),'Cellx'),P(f"{x['rows']:,}",'Cellx'),P(f"{x['invalid_rows']:,} ({x['invalid_fraction']:.4%})",'Cellx'),P(nice(x['raw_minus_released_kinematic_delta']['mJ1']['mae'],5),'Cellx'),P(nice(x['raw_minus_released_kinematic_delta']['dmJ']['mae'],5),'Cellx'),P(nice(x['raw_minus_released_kinematic_delta']['dRJJ']['mae'],2),'Cellx'),P(nice(x['raw_minus_released_kinematic_delta']['mJJ']['mae'],4),'Cellx')] for x in fv],[46*mm,16*mm,27*mm,22*mm,22*mm,17*mm,24*mm],6.2),Spacer(1,4),P('Across the raw validation sample, jet kinematics and angular separation agree closely with supplied values. Locally recomputed tau21 differs materially (MAE about 0.04–0.07) because the exact released FastJet-contrib configuration is not fully encoded here. This is why the authors’ supplied tau values define the primary experiment; the discrepancy is recorded rather than obscured.'),P('4. Methods and experiment matrix','H1x')]
    methodrows=[[P('ID','Cellx'),P('Method','Cellx'),P('Role','Cellx')],[P('M1','Cellx'),P('MLP autoencoder','Cellx'),P('Unsupervised reconstruction baseline','Cellx')],[P('M2','Cellx'),P('Isolation Forest','Cellx'),P('Classical unsupervised baseline','Cellx')],[P('M3','Cellx'),P('Deep SVDD','Cellx'),P('Unsupervised deep one-class model','Cellx')],[P('M4','Cellx'),P('Deep SAD','Cellx'),P('Semi-supervised scarce-label model','Cellx')],[P('M5','Cellx'),P('Supervised MLP','Cellx'),P('Fully supervised reference','Cellx')],[P('M6','Cellx'),P('Mass-aware Deep SVDD','Cellx'),P('Explicit mJJ leakage control','Cellx')]]
    story += [table(methodrows,[14*mm,55*mm,105*mm]),Spacer(1,5),P('The seed-42 grid varies signal contamination f_train over 0%, 0.1%, 0.5%, and 1% for M1–M3. Deep SAD varies contamination over 0%, 0.5%, and 1%, crossed with 10, 100, and 1,000 labeled signal events. M5 uses all labeled training signal available under its reference configuration; M6 tests direct mass access. Six headline configurations are additionally repeated with seeds 43 through 46 when present in the run table.'),P('Unsupervised models use background-only training statistics. Injected unlabeled signal examples are sampled without replacement with saved event IDs. Explicitly labeled signal events are disjoint from the unlabeled pool. Feature scaling is fitted on training data only. Deep SVDD center initialization excludes the labeled anomalies in the Deep SAD path.'),P('5. Training and evaluation protocol','H1x'),P('Models are trained on the training split only. Validation background scores determine the score thresholds for target background efficiencies of 10% and 1%; those thresholds are then frozen and applied to untouched test samples. Test labels are used only for final metrics. The split manifest, seeds, configuration grid, exact selected signal indices, scaler, and model checkpoint are saved alongside results.'),P('Reported ranking metrics include ROC AUC, background rejection at fixed signal efficiencies, and maximum significance-improvement characteristic. Topology transfer is measured by scoring the held-out 3-prong sample against the 2-prong background test sample. Mass dependence is summarized by score–mJJ association, Jensen–Shannon distance between the inclusive and selected background mass distributions, and binned acceptance plots.'),P('The fixed-window diagnostic uses the 3.3–3.7 TeV signal window with a 2.5–4.5 TeV sideband fit in 50 GeV bins. It reports local excess and fit uncertainty, plus a background-only Poisson bootstrap where available. The window is fixed; the statistic is local and must not be interpreted as a global look-elsewhere-corrected significance.')]
    story += [P('6. Measured results','H1x')]
    if charts:
        for c in charts:
            story += [Image(str(c),width=165*mm,height=87*mm),Spacer(1,3)]
    results42=df[df.seed==42]
    rows=[[P('Model / setting','Cellx'),P('f_train','Cellx'),P('k','Cellx'),P('AUC 2-prong','Cellx'),P('AUC 3-prong','Cellx'),P('JSD @10%','Cellx'),P('Score–mJJ','Cellx')]]
    for _,r in results42.sort_values(['model','f_train','k_labels']).iterrows():
        rows.append([P(escape(str(r.model).replace('_',' ')),'Cellx'),P(f'{r.f_train:g}%','Cellx'),P(str(int(r.k_labels)),'Cellx'),P(nice(r.roc_auc,3),'Cellx'),P(nice(r.roc_auc_3prong,3),'Cellx'),P(nice(r.mass_sculpting_jsd_at_10pct_bkg,3),'Cellx'),P(nice(r.score_mjj_dependence,3),'Cellx')])
    story += [table(rows,[42*mm,17*mm,12*mm,24*mm,24*mm,23*mm,25*mm],6.6),Spacer(1,5),P(f'The table lists {len(results42)} seed-42 configurations. All metrics are on simulated data. AUC measures ranking and is prevalence-independent; it is not a direct estimate of discovery power. The 3-prong metric measures cross-topology ranking, not training on the 3-prong signal.')]
    rep=ROOT/'reports/tables/replicate_summary.csv'
    if rep.exists():
        rd=pd.read_csv(rep,header=[0,1],index_col=[0,1,2]); rr=[[P('Model / setting','Cellx'),P('Seeds','Cellx'),P('AUC mean ± SD','Cellx'),P('3-prong AUC mean ± SD','Cellx')]]
        for key,row in rd.iterrows():
            try:
                if int(row[('roc_auc','count')]) < 2: continue
                rr.append([P(f'{key[0]} · f={float(key[1]):g} · k={int(key[2])}','Cellx'),P(str(int(row[('roc_auc','count')])),'Cellx'),P(f"{nice(row[('roc_auc','mean')])} ± {nice(row[('roc_auc','std')])}",'Cellx'),P(f"{nice(row[('roc_auc_3prong','mean')])} ± {nice(row[('roc_auc_3prong','std')])}",'Cellx')])
            except Exception: pass
        story.append(KeepTogether([P('Repeated-seed summary','H2x'),table(rr,[65*mm,18*mm,45*mm,48*mm],6.8)]))
    else: story.append(P('The repeat-seed aggregate has not been produced; no repeated-seed uncertainty is claimed.'))
    story += [P('7. Mass shaping and null diagnostics','H1x'),P('For each configuration, the evaluator saves inclusive and selected mass spectra, binned background acceptance, and mean score by mass. JSD compares normalized binned distributions. Lower JSD indicates closer shapes under that binning, but does not prove absence of mass dependence or guarantee a valid search procedure. Thresholds come from validation background; test mass plots are diagnostic.'),P('Background-only null trials repeatedly Poisson-resample the test background and refit the fixed-window sidebands. The current evaluator uses 50 pseudoexperiments per configuration. This small ensemble is a smoke-level diagnostic: its tail probability resolution is coarse (minimum 1/51), and it does not provide a calibrated global p-value. A null local significance near zero is expected when there is no injected excess.'),P('No bump claim is made. The bootstrap and local fit are reported to characterize the diagnostic and expose fit stability. Repeated-window searches, systematic uncertainties, control-region optimization, and real detector effects are outside the present scope.'),P('8. Reproducibility and artifacts','H1x'),P('From a clean Python 3.11 or 3.13 environment (the full run used Python 3.13), install the package and run python scripts/reproduce.py. The workflow verifies or downloads the four official data files, builds canonical feature tables, creates deterministic splits, fits the configured experiment grid, evaluates frozen thresholds, and writes the report inputs. Raw data are downloaded at runtime and excluded from version control and the Docker build context.'),P('Key outputs include data/manifest.yaml, data/splits/split_manifest.json, feature-validation JSON files, reports/tables/results.csv, reports/tables/run_manifest.json, replicate_summary.csv when completed, saved experiment sample IDs, MLflow records, model checkpoints, mass diagnostic figures, and this PDF. The report is regenerated with python scripts/build_report.py.'),P('Packaging includes an MIT license, CITATION.cff, dependency metadata, Dockerfile, and dataset provenance notes. Exact one-command reproducibility still depends on external Zenodo availability, sufficient disk space, and the specified Python packages.')]
    audit_intro=P('The implementation audit is intentionally separated from the PRD wish list. The following boundaries remain material to interpretation:','Bodyx')
    audit=[['Finding','Impact / status'],['No complete end-to-end three-prong training/evaluation protocol','Held-out topology is evaluated for scores; a matched 3-prong training regime is not part of the current matrix.'],['Released tau configuration is not reconstructed exactly','Canonical results use supplied high-level tau values; raw reclustering is validation only.'],['Five seeds for core headline settings; sweep mostly one seed','Small uncertainty sample; full grid robustness is not seed-averaged.'],['50-trial null bootstrap','Tail resolution and calibration are limited; no global significance claim.'],['Fixed-window diagnostic and simplified sideband fit','Not a complete bump-hunt search procedure or systematic model.'],['Simulation-only R&D samples','No detector-data or experimental-sensitivity claim.'],['No independent clean-room reproduction performed','Environment and data availability remain external dependencies.']]
    story.append(KeepTogether([P('9. Implementation audit and limitations','H1x'),audit_intro,table([[P(escape(x),'Cellx') for x in row] for row in audit],[60*mm,114*mm]),Spacer(1,5),P('The prior audit’s concrete fixes are reflected in the current sources: no dummy results are treated as evidence; subjectivity/tau features are computed or sourced explicitly rather than set to constants; train/validation/test roles are separated; thresholds are selected from validation; signal contamination is sampled with saved event IDs; null trials and mass diagnostics are emitted; and repository license/citation/packaging files are populated. Where experiments remain incomplete, the report states that directly.')]))
    story += [P('10. Conclusions and next steps','H1x'),P('The project now provides a usable, provenance-tracked benchmark implementation with meaningful measured outputs. The first complete comparison tests ranking performance, contamination response, scarce-label learning, topology transfer, and mass-shaping behavior in one reproducible workflow. Results do not justify a universal winner: the answer depends on the training regime and the diagnostic objective.'),P('To extend this into a stronger scientific study, increase the number of independent seeds across the full grid, calibrate the bump statistic with a larger predeclared null ensemble, predefine fit-quality and uncertainty acceptance criteria, and test an end-to-end topology-transfer regime. A clean environment reproduction and independent review of the released tau definition should precede any broader claims.'),P('11. References','H1x'),P('[1] LHC Olympics 2020 R&amp;D dataset, Zenodo record 6466204, version 5. https://doi.org/10.5281/zenodo.6466204'),P('[2] G. Kasieczka et al., “The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics,” arXiv:2101.08320 (2021). https://arxiv.org/abs/2101.08320'),P('[3] L. Ruff et al., “Deep One-Class Classification,” ICML 2018. https://proceedings.mlr.press/v80/ruff18a.html'),P('[4] L. Ruff et al., “Deep Semi-Supervised Anomaly Detection,” ICLR 2020. https://openreview.net/pdf?id=HkgH0TEYwH'),P('[5] Project protocol and requirements: PRD_final.md in this repository.')]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.saveState(); w,h=A4; canvas.setStrokeColor(colors.HexColor('#d1dbe2')); canvas.line(18*mm,15*mm,w-18*mm,15*mm)
        canvas.setFont('Helvetica',7.5); canvas.setFillColor(colors.HexColor('#536575')); canvas.drawString(18*mm,10*mm,'LHCO 2020 R&D · Final project report · Simulation-only')
        canvas.drawRightString(w-18*mm,10*mm,str(doc.page)); canvas.restoreState()
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=21*mm,title='LHC Olympics Anomaly Detection — Final Project Report',author='Mithil Hardik Astik',subject='Reproducible simulation-only benchmark and implementation audit')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    baseline=df[(df.seed==42)&(df.f_train==0)&(df.k_labels==0)].sort_values('roc_auc',ascending=False)
    note=['# Technical Note: Unsupervised and Semi-Supervised Anomaly Detection on the LHC Olympics 2020 R&D Dataset','',
          '## Status and scope','',f'The simulation-only benchmark contains {len(df)} unique result configurations over seeds {", ".join(map(str,seeds))}. The full contamination/label-budget matrix is seed 42; repeated seeds 43 through 46 cover six headline configurations. No discovery or real-data sensitivity claim is made.','',
          '## Methods and data','', 'M1 is an MLP autoencoder; M2 Isolation Forest; M3 Deep SVDD; M4 Deep SAD; M5 a supervised MLP reference; M6 a mass-aware Deep SVDD leakage control. Inputs are mJ1, dmJ, tau21 for each leading jet, and dRJJ. mJJ is excluded from M1–M5 and included only in M6. Official Zenodo v5 high-level features are canonical; raw anti-kT/exclusive-kT validation is a deterministic 128-event check. The local tau axes do not exactly reproduce the supplied FastJet-contrib convention.','',
          '## Measured seed-42 headline results','', '| Model | f_train | k | 2-prong AUC | 3-prong AUC | JSD at 10% |','|---|---:|---:|---:|---:|---:|']
    note_specs=[('M1_Autoencoder',0.0,0),('M2_IsolationForest',0.0,0),('M3_DeepSVDD',0.0,0),('M4_DeepSAD',0.5,100),('M5_Supervised',0.0,1000),('M6_MassAware',0.0,0)]
    for model,f,k in note_specs:
        selected=df[(df.model==model)&(df.seed==42)&(df.f_train==f)&(df.k_labels==k)]
        if len(selected):
            r=selected.iloc[0]
            note.append(f"| {r.model} | {r.f_train:g}% | {int(r.k_labels)} | {r.roc_auc:.4f} | {r.roc_auc_3prong:.4f} | {r.mass_sculpting_jsd_at_10pct_bkg:.4f} |")
    note += ['', 'Metrics are ranking/shape diagnostics on simulated test data. The held-out 3-prong sample is never used to train these configurations. Thresholds are selected on validation background and frozen for test evaluation. The fixed-window sideband diagnostic is local; its 50-trial Poisson bootstrap has coarse tail resolution and is not a global p-value.','',
             '## Reproducibility and limitations','', 'Run `python -m pip install -e ".[dev]"` followed by `python scripts/reproduce.py`. The run verifies the official data files, builds features and deterministic splits, evaluates the grid, and writes model/data/run metadata under `data/`, `models/`, and `reports/`. Regenerate the PDF with `python scripts/build_report.py`.','', 'Remaining limitations include simulation-only scope, the released/local tau definition mismatch, five seeds for headline runs, a seed-42 full grid, the limited null bootstrap ensemble, simplified fixed-window sideband fitting, and lack of a clean-room independent reproduction. See the PDF report and `PRD_final.md` for full protocol and audit detail.','']
    (ROOT/'reports'/'benchmark_technical_note.md').write_text('\n'.join(note),encoding='utf-8')
    print(f'Wrote {OUT} ({OUT.stat().st_size:,} bytes), based on {len(df)} result rows and {len(charts)} charts.')

if __name__=='__main__': build()




