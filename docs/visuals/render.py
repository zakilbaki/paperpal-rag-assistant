"""Rebuild documentation SVGs and recorded-result plots from visual_spec.json.

Diagrams use Python's standard library. Plot generation requires matplotlib.
Run from any directory: python docs/visuals/render.py
"""
from pathlib import Path
import json
from html import escape
import re, math

HERE = Path(__file__).resolve().parent

def svg_document(spec):
    w, h = spec['width'], spec['height']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(spec["title"])}</title><desc id="desc">{escape(spec["description"])}</desc>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#64748b"/></marker></defs>',
             f'<rect width="{w}" height="{h}" fill="#f8fafc"/>',
             '<g font-family="Arial, Helvetica, sans-serif">']
    for op in spec['elements']:
        kind = op['kind']; a = op.copy(); del a['kind']
        if kind == 'text':
            content = escape(a.pop('text'))
            attr = ' '.join(f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k,v in a.items())
            parts.append(f'<text {attr}>{content}</text>')
        else:
            arrow = a.pop('marker_end', None)
            attr = ' '.join(f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k,v in a.items())
            parts.append(f'<{kind} {attr}/>')
            if arrow:
                coords = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', a['d'])]
                x0,y0,x,y=coords[-4:];dx=x-x0;dy=y-y0;l=math.hypot(dx,dy);ux=dx/l;uy=dy/l
                points=f'{x},{y} {x-8*ux-4*uy},{y-8*uy+4*ux} {x-8*ux+4*uy},{y-8*uy-4*ux}'
                parts.append(f'<polygon points="{points}" fill="#64748b"/>')
    return '\n'.join(parts + ['</g></svg>'])

def plot(spec, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelcolor':'#334155',
      'text.color':'#0f172a','axes.edgecolor':'#cbd5e1','xtick.color':'#475569','ytick.color':'#475569',
      'figure.facecolor':'#f8fafc','axes.facecolor':'#ffffff','svg.hashsalt':'portfolio-2026',
      'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(10.8,5.8))
    fig.subplots_adjust(left=.13,right=.95,top=.76,bottom=.23)
    fig.text(.07,.94,spec['eyebrow'].upper(),color='#0f766e',fontsize=10,weight='bold')
    fig.text(.07,.875,spec['title'],fontsize=21,weight='bold')
    fig.text(.07,.815,spec['subtitle'],fontsize=11,color='#475569')
    if spec['kind']=='confusion':
        a=np.array(spec['matrix']); norm=a/a.sum(axis=1,keepdims=True)
        ax.imshow(norm,cmap=ListedColormap(['#eef2f6','#a7d8d0','#0f766e']),vmin=0,vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j,i-.09,f'{a[i,j]:,}',ha='center',va='center',fontsize=23,weight='bold',color='white' if norm[i,j]>.67 else '#0f172a')
                ax.text(j,i+.16,f'{norm[i,j]:.2%}',ha='center',va='center',fontsize=10,color='white' if norm[i,j]>.67 else '#334155')
        ax.set_xticks([0,1],spec['labels']);ax.set_yticks([0,1],spec['labels'])
        ax.set_xlabel('Predicted class',labelpad=9);ax.set_ylabel('Actual class')
        ax.tick_params(length=0);ax.set_aspect('equal')
        for spine in ax.spines.values():spine.set_visible(False)
    elif spec['kind']=='operating':
        p=np.array(spec['precision'])*100;r=np.array(spec['recall'])*100
        ax.plot(r,p,'o-',color='#0f766e',lw=2,markersize=8)
        for i,(xx,yy,t) in enumerate(zip(r,p,spec['threshold'])):
            ax.annotate(f't = {t:.4f}',(xx,yy),xytext=(8,7 if i%2 else -18),textcoords='offset points',fontsize=10)
        ax.set(xlim=(65,82),ylim=(75,100),xlabel='Recall (%)',ylabel='Precision (%)')
        ax.grid(alpha=.22);ax.set_axisbelow(True)
    elif spec['kind']=='comparison':
        y=np.arange(len(spec['models'])); colors=['#0f766e','#6366a5']
        for k,series in enumerate(spec['series']):
            ax.scatter(series['values'],y+(k-.5)*.17,label=series['label'],s=80,color=colors[k],zorder=3)
            for yy,v in zip(y+(k-.5)*.17,series['values']):ax.annotate(f'{v:.2f}',(v,yy),xytext=(7,0),textcoords='offset points',va='center',fontsize=10)
        ax.set_yticks(y,spec['models']);ax.invert_yaxis();ax.set(xlim=spec['xlim'],xlabel=spec['xlabel'])
        ax.grid(axis='x',alpha=.22);ax.legend(loc='upper left',bbox_to_anchor=(0,-.24),frameon=False,ncol=2)
        fig.subplots_adjust(left=.24,bottom=.29)
    fig.text(.07,.075,spec['footnote'],fontsize=10,color='#475569')
    fig.text(.07,.035,spec['source'],fontsize=9,color='#64748b')
    fig.savefig(output,metadata={'Date':None},facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix('.png'),dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    data=json.loads((HERE/'visual_spec.json').read_text())
    out=HERE.parent/'assets';out.mkdir(exist_ok=True)
    for d in data['diagrams']:(out/d['filename']).write_text(svg_document(d))
    for p in data.get('plots',[]):plot(p,out/p['filename'])

if __name__=='__main__':main()
