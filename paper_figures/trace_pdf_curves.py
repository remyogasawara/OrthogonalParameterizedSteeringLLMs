"""Recover the plotted (alpha, avg_score) values of the two before/after-clipping figure PDFs
(clipping_cropped.pdf and alpha_v_param.pdf, vector matplotlib output) -> data/fig2_recovered.csv and
data/fig4_recovered.csv, which make_figs.fig2 / fig3_4_merged replot. The experiment pickles behind those two
figures were not archived, so their curves are traced from the figure PDFs themselves.

Input: the page trace of each PDF from MuPDF,
    mutool draw -F trace -o fig2_trace.xml clipping_cropped.pdf
    mutool draw -F trace -o fig4_trace.xml alpha_v_param.pdf
run in the same directory. Axes are calibrated from the tick marks (x ticks -2.0..2.0 step 0.5, y ticks listed
per panel below); every recovered alpha lands on the 0.1 grid. Two vertices are missing from the traced
before-clipping path (power at alpha 1.2, wealth at alpha 0.0); in data/fig2_recovered.csv they were added by hand
from the centres of the plotted circle markers (3.358 and -1.672). The script itself writes only the traced
vertices, so its output has two rows fewer than the committed file.
"""
import re, sys, csv, collections
COLS={'.1215686 .4666667 .7058824':'coordinate-other-ais','1 .4980392 .05490196':'corrigible-neutral-HHH',
      '.172549 .627451 .172549':'myopic-reward','.8392158 .1529412 .1568628':'survival-instinct',
      '.5803922 .4039216 .7411765':'power-seeking-inclination','.5490196 .3372549 .2941177':'wealth-seeking-inclination'}
def T(tr,p):
    a,b,c,d,e,f=map(float,tr.split()); x,y=p; return (a*x+c*y+e, b*x+d*y+f)
def run(trace, yticks_per_panel, panel_names, out_csv):
    txt=open(trace).read()
    curves=[]; xt=[]; yt=[]
    for m in re.finditer(r'<(stroke_path|fill_path) ([^>]*)>(.*?)</\1>', txt, re.S):
        if 'curveto' in m.group(3): continue
        a=dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
        P=[T(a['transform'],(float(x),float(y))) for x,y in re.findall(r'<(?:moveto|lineto) x="([^"]+)" y="([^"]+)"', m.group(3))]
        col=a.get('color')
        if m.group(1)=='stroke_path' and col in COLS and len(P)>=30: curves.append((COLS[col],P))
        if m.group(1)=='stroke_path' and col=='0 0 0' and len(P)==2:
            w=abs(P[0][0]-P[1][0]); h=abs(P[0][1]-P[1][1])
            if w<1e-3 and 1.3<h<1.6: xt.append(P[0])
            if h<1e-3 and 1.3<w<1.6: yt.append(P[0])
    # split into panels by x position of curve start
    starts=sorted({round(P[0][0]) for _,P in curves})
    split=(min(starts)+max(starts))/2
    rows=[]
    for k,(name,yvals) in enumerate(zip(panel_names,yticks_per_panel)):
        left = k==0
        cs=[(n,P) for n,P in curves if (P[0][0]<split)==left]
        xs=sorted({round(p[0],3) for p in xt if (p[0]<split)==left})
        ys=sorted({round(p[1],3) for p in yt if (p[0]<split)==left}, reverse=True)  # page y grows downward? check both
        # x calibration: 9 ticks = -2.0..2.0
        import statistics
        step=statistics.median([b-a for a,b in zip(xs,xs[1:])])   # page units per 0.5 alpha
        ax=2*step
        bx=min(P[0][0] for _,P in cs)                               # first vertex = alpha -2.0
        print(f"  [{name}] x-ticks found={len(xs)} first tick={xs[0]:.3f} first vertex={bx:.3f} step/0.5={step:.3f}")
        # y calibration: map sorted tick positions to tick values (orientation from data)
        assert len(ys)==len(yvals), (ys,yvals)
        ys_sorted=sorted(ys)
        # page y increases downward in mutool trace (origin top-left) -> smallest y is largest value
        vals=sorted(yvals, reverse=True)
        ay=(vals[-1]-vals[0])/(ys_sorted[-1]-ys_sorted[0]); by=vals[0]-ay*ys_sorted[0]
        for n,P in cs:
            for (px,py) in P:
                alpha=-2.0+(px-bx)/ax
                rows.append((name,n,round(alpha,3),round(ay*py+by,4)))
    with open(out_csv,'w',newline='') as f:
        w=csv.writer(f, lineterminator='\n'); w.writerow(['panel','behavior','alpha','avg_score']); w.writerows(rows)
    return rows
if __name__=='__main__':
    import os
    HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
    r2=run('fig2_trace.xml',[[-2,0,2,4,6,8],[-2,0,2,4,6,8]],['fig2_before_clipping','fig2_after_clipping'],os.path.join(D,'fig2_recovered.csv'))
    r4=run('fig4_trace.xml',[[-2,0,2,4,6,8],[-2,0,2,4,6,8]],['fig4_alpha_iterative','fig4_parameterized'],os.path.join(D,'fig4_recovered.csv'))
    for rows in (r2,r4):
        d=collections.defaultdict(dict)
        for p,b,a,v in rows: d[(p,b)][a]=v
        for (p,b),m in sorted(d.items()):
            ks=sorted(m); mx=max(m,key=m.get); mn=min(m,key=m.get)
            print(f"{p:22s} {b:28s} n={len(m):2d} a0={ks[0]:+.2f} a_end={ks[-1]:+.2f} f(-2)={m[ks[0]]:+.3f} max={m[mx]:+.3f}@{mx:+.2f} min={m[mn]:+.3f}@{mn:+.2f}")
