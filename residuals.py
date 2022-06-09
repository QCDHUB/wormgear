import sys,os
import numpy as np
from scipy.integrate import cumtrapz
from obslib.dihadron.reader import READER
from tools.residuals import _RESIDUALS
from tools.config import conf
from tools.tools import load,save
import time
from scipy.interpolate import griddata
from qpdlib.qpdcalc import QPDCALC

class RESIDUALS(_RESIDUALS):
  
    def __init__(self):
 
        self.reaction='wormgear' 
        self.tabs=conf['wormgear tabs']
        if 'g1T'  in conf: self.g1T  = conf['diffpippim']
        self.setup()
        self.flavs = ['u','d']

        if 'lhapdf_pdf'   in conf: self.lhapdf_pdf = conf['lhapdf_pdf']
        else:                    self.lhapdf_pdf = 'JAM22'
        if 'lhapdf_ffpi'  in conf: self.lhapdf_ffpi  = conf['lhapdf_ffpi']
        else:                      self.lhapdf_ffpi  = 'JAM22'
        if 'lhapdf_ffhad' in conf: self.lhapdf_ffhad = conf['lhapdf_ffhad']
        else:                      self.lhapdf_ffhad = 'JAM22'

    def get_f1(self,x,Q2):

        #--get PDFs
        f1 = {_:[] for _ in self.flavs}
        os.environ['LHAPDF_DATA_PATH'] = 'obslib/wormgear/lhapdf'
        QCF = lhapdf.mkPDFs(self.lhapdf_pdf)

        f1['u'] = np.array([QCF.xfxQ2(2,x[i],Q2[i])/x[i] for i in range(len(x))])
        f1['d'] = np.array([QCF.xfxQ2(1,x[i],Q2[i])/x[i] for i in range(len(x))])

        return f1

    def get_D1(self,had,z,Q2):

        #--get FFss
        D1 = {_:[] for _ in self.flavs}
        os.environ['LHAPDF_DATA_PATH'] = 'obslib/wormgear/lhapdf'
        if had in ['pi+', 'pi-', 'pi0']:
            QCF = lhapdf.mkPDFs(self.lhapdf_ff)
        if had in ['h+', 'h-']:
            QCF = lhapdf.mkPDFs(self.lhapdf_had)

        #--get mean value
        QCF = QCF[0]

        fav = np.array([QCF.xfxQ2(2,z[i],Q2[i])/z[i] for i in range(len(z))])
        unf = np.array([QCF.xfxQ2(1,z[i],Q2[i])/z[i] for i in range(len(z))])

        if had=='pi+' or had=='h+':
            D1['u'] = fav
            D1['d'] = unf
        elif had=='pi-' or had=='h-':
            D1['u'] = unf
            D1['d'] = fav
        elif had=='pi0':
            D1['u'] = (fav + unf)/2.0
            D1['d'] = (fav + unf)/2.0

        return D1

    def get_A_LT(self,idx,had):
    

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        x     = self.tabs[idx]['x']
        Q2    = self.tabs[idx]['Q2']
        z     = self.tabs[idx]['z']
        M     = self.tabs[idx]['M']
        PhT   = self.tabs[idx]['PhT']
        tar   = self.tabs[idx]['tar'][0]
        had    = self.tabs[idx]['hadrons'][0].strip()

        if tar=='p' or tar=='n':
            M = conf['aux'].M

    
        #--get PDFs and FFs from LHAPDF
        f1 = self.get_f1(x,Q2)
        D1 = self.get_D1(had,z,Q2)

        #--get g1T from qcdlib
        g1T = self.g1T.get_xf()


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

            lambda_f1  = z**2*kp2_f1[flav]  + Pp2_D1[flav]
            lambda_g1T = z**2*kp2_g1T[flav] + Pp2_D1[flav]

            g_f1  = np.exp(-PhT**2/lambda_f1) /(np.pi*lambda_f1)
            g_g1T = np.exp(-PhT**2/lambda_g1T)/(np.pi*lambda_g1T)

            FUU += x*e**2*f1[flav]*D1[flav]*g_f1

            FLT += 2*M*x*z*PhT*e**2*g1T[flav]*D1[flav]*g_g1T/lambda_g1T

        
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
 






 
