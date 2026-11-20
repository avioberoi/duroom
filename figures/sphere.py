"""Sphere display with exact polar angles and projected azimuths; a 3D view of high-dimensional vectors, not an isometric embedding. Only the angle to the mean direction is preserved; azimuth uses PC1 and PC2.
Usage: python sphere.py"""
from _panels import *
from matplotlib.patches import Circle, FancyArrowPatch

d=load_data('measured_sphere')
xy=np.array(d['xy']);theta=np.deg2rad(d['angles']);phi=np.arctan2(xy[:,1],xy[:,0])
xyz=np.column_stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
p=d['percentiles']
assert np.allclose(np.linalg.norm(xyz,axis=1),1)
assert np.allclose(np.rad2deg(np.arccos(xyz[:,2])),d['angles'])

def project(points,elevation):
    x,y,z=np.asarray(points).T
    e=np.deg2rad(elevation)
    return np.column_stack([x,z*np.cos(e)-y*np.sin(e),y*np.cos(e)+z*np.sin(e)])

def latitude(degrees):
    t=np.deg2rad(degrees);a=np.linspace(0,2*np.pi,361)
    return np.column_stack([np.sin(t)*np.cos(a),np.sin(t)*np.sin(a),np.full_like(a,np.cos(t))])

for mobile in (False,True):
    fig,axes=plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(3.9,8.1) if mobile else (8.8,4.8))
    for ax,elev,title in zip(axes.flat,(30,90),('Works on the unit sphere','Looking down the pole' if mobile else 'Looking down the average direction')):
        ax.set_aspect('equal');ax.set_xlim(-1.48,1.48);ax.set_ylim(-1.3,1.4);ax.axis('off')
        ax.add_patch(Circle((0,0),1,fill=False,edgecolor=HAIR,lw=.9))
        # Equator and one meridian establish the sphere's volume.
        if elev!=90:
            for ring in [latitude(90),np.column_stack([np.zeros(361),np.sin(np.linspace(0,2*np.pi,361)),np.cos(np.linspace(0,2*np.pi,361))])]:
                r=project(ring,elev);ax.plot(r[:,0],r[:,1],color=HAIR,lw=.6,ls=(0,(3,3)),zorder=0)
        for percentile,color,ls in [('50',MUTE,'-'),('99',ACCENT,(0,(4,3)))]:
            r=project(latitude(p[percentile]),elev);ax.plot(r[:,0],r[:,1],color=color,lw=.8,ls=ls,zorder=2)
        points3=project(xyz,elev);order=np.argsort(points3[:,2])
        ax.scatter(points3[order,0],points3[order,1],s=2.5 if mobile else 3,color=INK,alpha=.42,linewidth=0,zorder=3)
        pole=project([[0,0,1]],elev)[0,:2]
        mean=project([[0,0,d['mean_norm']]],elev)[0,:2]
        if elev!=90:
            # The mean is inside the sphere, distinct from the unit pole.
            ax.plot([mean[0],pole[0]],[mean[1],pole[1]],color=ACCENT,lw=.7,ls=(0,(2,2)),zorder=4)
            ax.add_patch(FancyArrowPatch((0,0),mean,arrowstyle='-|>',mutation_scale=10,color=ACCENT,lw=1.2,zorder=5))
            ax.scatter(*mean,s=24,color=ACCENT,zorder=6)
            ax.scatter(0,0,s=10,color=MUTE,zorder=5)
            ax.text(.10,-.10,'Origin',fontsize=10,color=MUTE)
            ax.annotate('Average vector\ninside the sphere',xy=mean,xytext=(-1.43,-.78),fontsize=11,ha='left',va='top',color=ACCENT,
                        arrowprops={'arrowstyle':'-','color':ACCENT,'lw':.7,'connectionstyle':'arc3,rad=-.25'})
            ax.annotate('Average direction',xy=pole,xytext=(0,1.22),fontsize=11,ha='center',va='bottom',color=MUTE,
                        arrowprops={'arrowstyle':'-','color':MUTE,'lw':.7})
        else:
            ax.scatter(0,0,s=14,color=ACCENT,zorder=5)
            ax.text(0,.11,'Pole',ha='center',va='bottom',fontsize=11,color=ACCENT)
            ax.text(0,-.14,'No work here',ha='center',va='top',fontsize=10,color=MUTE)
        panel_label(ax,title,mobile)
    fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.19 if mobile else .20,wspace=.08,hspace=.20)
    legend=f'Median {p["50"]:.1f}\u00b0 (solid)'+ ('\n' if mobile else '   |   ')+f'99% within {p["99"]:.1f}\u00b0 (dashed)'
    extent=f'Nearest {p["0"]:.1f}\u00b0; farthest {p["100"]:.1f}\u00b0'+ ('\n' if mobile else ' ')+'from the average direction.'
    fig.text(.5,.10 if mobile else .105,legend,ha='center',fontsize=10,color=MUTE,linespacing=1.4)
    fig.text(.5,.035 if mobile else .055,extent,ha='center',fontsize=10,color=MUTE,linespacing=1.4)
    finish(fig,'sphere',mobile)
