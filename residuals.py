import sys,os
import numpy as np
from scipy.integrate import cumtrapz
from obslib.dihadron.reader import READER
from tools.residuals import _RESIDUALS
from tools.config import conf
from tools.tools import load,save
import time
from scipy.interpolate import griddata
import lhapdf

class RESIDUALS(_RESIDUALS):
  
    def __init__(self):
 
        self.reaction='wormgear' 
        self.tabs=conf['wormgear tabs']
        if 'g1T'  in conf: self.g1T  = conf['diffpippim']
        self.setup()
        self.flavs = ['u','d']

        if 'lhapdf_pdf'   in conf: self.lhapdf_pdf = conf['lhapdf_pdf']
        else:                    self.lhapdf_pdf = 'JAM22-PDF_proton_nlo'
        if 'lhapdf_ffpi'  in conf: self.lhapdf_ffpi  = conf['lhapdf_ffpi']
        else:                      self.lhapdf_ffpi  = 'JAM22-FF_pion_nlo'
        if 'lhapdf_ffhad' in conf: self.lhapdf_ffhad = conf['lhapdf_ffhad']
        else:                      self.lhapdf_ffhad = 'JAM22-FF_hadron_nlo'

        os.environ['LHAPDF_DATA_PATH'] = 'obslib/wormgear/lhapdf'
        self.setup_f1()
        self.setup_D1()

    def setup_f1(self):

        self.f1 = {}

        for idx in self.tabs:

            self.f1[idx] = {_:[] for _ in self.flavs}

            X     = self.tabs[idx]['X']
            Q2    = self.tabs[idx]['Q2']

            #--get PDFs
            QCF = lhapdf.mkPDFs(self.lhapdf_pdf)

            self.f1[idx]['u'] = np.array([QCF.xfxQ2(2,x[i],Q2[i])/x[i] for i in range(len(x))])
            self.f1[idx]['d'] = np.array([QCF.xfxQ2(1,x[i],Q2[i])/x[i] for i in range(len(x))])

    def setup_D1(self):

        for idx in self.tabs:

            self.D1[idx] = {_:[] for _ in self.flavs}

            Z     = self.tabs[idx]['Z']
            Q2    = self.tabs[idx]['Q2']
            had   = self.tabs[idx]['hadron'][0].strip()

            #--get FFs
            if had in ['pi+', 'pi-', 'pi0']:
                QCF = lhapdf.mkPDFs(self.lhapdf_ff)
            if had in ['h+', 'h-']:
                QCF = lhapdf.mkPDFs(self.lhapdf_had)

            #--get mean value
            QCF = QCF[0]

            fav = np.array([QCF.xfxQ2(2,z[i],Q2[i])/z[i] for i in range(len(z))])
            unf = np.array([QCF.xfxQ2(1,z[i],Q2[i])/z[i] for i in range(len(z))])

            if had=='pi+' or had=='h+':
                self.D1[idx]['u'] = fav
                self.D1[idx]['d'] = unf
            elif had=='pi-' or had=='h-':
                self.D1[idx]['u'] = unf
                self.D1[idx]['d'] = fav
            elif had=='pi0':
                self.D1[idx]['u'] = (fav + unf)/2.0
                self.D1[idx]['d'] = (fav + unf)/2.0

    def get_A_LT(self,idx,had):
    

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        X     = self.tabs[idx]['X']
        Q2    = self.tabs[idx]['Q2']
        Z     = self.tabs[idx]['Z']
        PhT   = self.tabs[idx]['PhT']
        tar   = self.tabs[idx]['tar'][0]
        had    = self.tabs[idx]['hadron'][0].strip()

        if tar=='p' or tar=='n':
            M = conf['aux'].M

    
        #--get PDFs and FFs from LHAPDF
        f1 = self.f1[idx]
        D1 = self.D1[idx]

        #--get g1T from qcdlib
        g1T = {}
        g1T['u'] = np.array([self.g1T.get_xf(X[i],Q2[i],'u')/X[i] for i in range(len(X))])
        g1T['d'] = np.array([self.g1T.get_xf(X[i],Q2[i],'d')/X[i] for i in range(len(X))])

        #--isospin symmetry for neutron
        if tar=='n':
            u,d = f1['u'],f1['d']
            f1['u'],f1['d'] = d,u

            u,d = g1T['u'],g1T['d']
            g1T['u'],g1T['d'] = d,u

        #--widths
        kp2_g1T, kp2_f1, Pp2_D1 = {},{},{}

        #--taken from inspirehep.net/literature/1781484
        kp2_f1['u'] = 0.53
        kp2_f1['d'] = 0.53

        #--taken from inspirehep.net/literature/828163
        kp2_g1T['u'] = 0.40
        kp2_g1T['d'] = 0.40

        #--taken from inspirehep.net/literature/1781484
        Pp2_D1['u'] = 0.124
        Pp2_D1['d'] = 0.145


        FUU, FLT = 0.0, 0.0
        for flav in self.flavs:
            if   flav in ['u']: e = ep
            elif flav in ['d']: e = ed

            lambda_f1  = Z**2*kp2_f1[flav]  + Pp2_D1[flav]
            lambda_g1T = Z**2*kp2_g1T[flav] + Pp2_D1[flav]

            g_f1  = np.exp(-PhT**2/lambda_f1) /(np.pi*lambda_f1)
            g_g1T = np.exp(-PhT**2/lambda_g1T)/(np.pi*lambda_g1T)

            FUU += X*e**2*f1[flav]*D1[flav]*g_f1

            FLT += 2*M*X*Z*PhT*e**2*g1T[flav]*D1[flav]*g_g1T/lambda_g1T

        
        thy = FLT/FUU
 
        return thy

    def get_theory(self):
        
        for idx in self.tabs:
            obs    = self.tabs[idx]['obs'][0].strip()

            if obs=='A_LTcos(phi_h-phi_S)':
                self.get_A_LT(idx,had)

            self.tabs[idx]['thy'] = thy
    
    def gen_report(self,verb=1,level=1):
        """
        verb = 0: Do not print on screen. Only return list of strings
        verv = 1: print on screen the report
        level= 0: only the total chi2s
        level= 1: include point by point 
        """
          
        L=[]
  
        if len(self.tabs.keys())!=0:
            L.append('reaction: dihadron')
            for f in conf['datasets']['dihadron']['filters']:
                L.append('filters: %s'%f)
  
            L.append('%7s %3s %20s %5s %10s %10s %10s %10s %10s'%('idx','tar','col','npts','chi2','chi2-npts','chi2/npts','rchi2','nchi2'))
            for k in self.tabs:
                if len(self.tabs[k])==0: continue 
                res=self.tabs[k]['residuals']
  
                rres=[]
                for c in conf['rparams']['dihadron'][k]:
                    rres.append(conf['rparams']['dihadron'][k][c]['value'])
                rres=np.array(rres)
  
                if k in conf['datasets']['dihadron']['norm']:
                    norm=conf['datasets']['dihadron']['norm'][k]
                    nres=(norm['value']-1)/norm['dN']
                else:
                    nres=0
  
                chi2=np.sum(res**2)
                rchi2=np.sum(rres**2)
                nchi2=nres**2
                if 'target' in self.tabs[k]: tar=self.tabs[k]['target'][0]
                else: tar = '-'
                col=self.tabs[k]['col'][0].split()[0]
                npts=res.size
                L.append('%7d %3s %20s %5d %10.2f %10.2f %10.2f %10.2f %10.2f'%(k,tar,col,npts,chi2,chi2-npts,chi2/npts,rchi2,nchi2))
  
            if level==1:
              L.append('-'*100)  
              for k in self.tabs:
                  if len(self.tabs[k]['value'])==0: continue 
                  if k in conf['datasets']['SU23']['norm']:
                      norm=conf['datasets']['SU23']['norm'][k]
                      nres=(norm['value']-1)/norm['dN']
                      norm=norm['value']
                  else:
                      norm=1.0
                      nres=0
                  for i in range(len(self.tabs[k]['value'])):
                      x     = self.tabs[k]['X'][i]
                      Q2    = self.tabs[k]['Q2'][i]
                      res   = self.tabs[k]['residuals'][i]
                      thy   = self.tabs[k]['thy'][i]
                      exp   = self.tabs[k]['value'][i]
                      alpha = self.tabs[k]['alpha'][i]
                      rres  = self.tabs[k]['r-residuals'][i]
                      col   = self.tabs[k]['col'][i]
                      shift = self.tabs[k]['shift'][i]
                      if 'target' in self.tabs[k]: tar   = self.tabs[k]['target'][i]
                      else: tar = '-'
                      msg='%d col=%7s, tar=%5s, x=%10.3e, Q2=%10.3e, exp=%10.3e, alpha=%10.3e, thy=%10.3e, shift=%10.3e, chi2=%10.3e, res=%10.3e, norm=%10.3e, '
                      L.append(msg%(k,col,tar,x,Q2,exp,alpha,thy,shift,res**2,res,norm))
  
        if verb==0:
            return L
        elif verb==1:
            for l in L: print(l)
            return L
 






 
