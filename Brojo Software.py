from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox
import soundfile as sf
import scipy.signal as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
import csv



def func_loadplan():
    
    list.clear(planxy)
    list.clear(planxz)
    list.clear(planyz)

    name = filedialog.askopenfilename(title="Load XY Plan", filetypes=[("PNG Image", "*.png")])
    if len(name)!=0:
        list.append(planxy,mpimg.imread(name))   
    name = filedialog.askopenfilename(title="Load XZ Plan", filetypes=[("PNG Image", "*.png")])
    if len(name)!=0:
        list.append(planxz,mpimg.imread(name)) 
    name = filedialog.askopenfilename(title="Load YZ Plan", filetypes=[("PNG Image", "*.png")])
    if len(name)!=0:
        list.append(planyz,mpimg.imread(name)) 
    
    return


def func_load(form):
    
    w = Tk()
    w.title("")
    w.resizable(False,False)
    lbl = Label(w, text="Choose measurement type")
    lbl.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
    btnsweep = Button(w, text="Sine Sweep", width=14, command=lambda: (w.destroy(),func_load2(form,"sweep")))
    btnsweep.grid(row=1,column=0, padx=10, pady=10)
    btnir = Button(w, text="Impulse Response", width=14, command=lambda: (w.destroy(),func_load2(form,"ir")))
    btnir.grid(row=1,column=1, padx=10, pady=10)
    w.mainloop()
    
    return


def func_load2(form,mestype):
    try:
        global fs
        showhidecontrols(0)
        if form=="A":
            btn_loadA["text"] = "Loading"
            FLUname = filedialog.askopenfilename(title="Load Front-Left-Up measurment (FLU)", filetypes=[("WAV Audio", "*.wav")])
            if len(FLUname)==0:
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            FRDname = filedialog.askopenfilename(title="Load Front-Right-Down measurment (FRD)", filetypes=[("WAV Audio", "*.wav")])
            if len(FRDname)==0:
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            BLDname = filedialog.askopenfilename(title="Load Back-Left-Down measurment (BLD)", filetypes=[("WAV Audio", "*.wav")])
            if len(BLDname)==0:
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            BRUname = filedialog.askopenfilename(title="Load Back-Right-Up measurment (BRU)", filetypes=[("WAV Audio", "*.wav")])
            if len(BRUname)==0:
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            [FLU,FLUfs] = sf.read(FLUname)
            [FRD,FRDfs] = sf.read(FRDname)
            [BLD,BLDfs] = sf.read(BLDname)
            [BRU,BRUfs] = sf.read(BRUname)
            if len(FLU)!=len(FRD) or len(FLU)!=len(BLD) or len(FLU)!=len(BRU):
                messagebox.showerror("Error", message="Signals length mismatch")
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            if len(np.shape(FLU))!=1 or len(np.shape(FRD))!=1 or len(np.shape(BLD))!=1 or len(np.shape(BRU))!=1:
                messagebox.showerror("Error", message="Signals must have 1 channel")
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            if FLUfs!=FRDfs or FLUfs!=BLDfs or FLUfs!=BRUfs:
                messagebox.showerror("Error", message="Signals sample rate mismatch")
                btn_loadA["text"] = "Load A format"
                showhidecontrols(1)
                return
            if mestype=="sweep":
                InvFiltname= filedialog.askopenfilename(title="Load Inverse Filter", filetypes=[("WAV Audio", "*.wav")])
                if len(InvFiltname)==0:
                    btn_loadA["text"] = "Load A format"
                    showhidecontrols(1)
                    return
                [InvFilt,InvFiltfs] = sf.read(InvFiltname)
                if FLUfs!=InvFiltfs:
                    messagebox.showerror("Error", message="Inverse filter sample rate and data sample rate mismatch")    
                    btn_loadA["text"] = "Load A format"
                    showhidecontrols(1)
                    return
                if len(np.shape(InvFilt))!=1:
                    messagebox.showerror("Error", message="Inverse filter must have 1 channel")    
                    btn_loadA["text"] = "Load A format"
                    showhidecontrols(1)
                    return
                FLU = sc.fftconvolve(InvFilt, FLU, mode="valid")
                FRD = sc.fftconvolve(InvFilt, FRD, mode="valid")
                BLD = sc.fftconvolve(InvFilt, BLD, mode="valid")
                BRU = sc.fftconvolve(InvFilt, BRU, mode="valid")
            fs = FLUfs
            W = FLU + FRD + BLD + BRU
            X = FLU + FRD - BLD - BRU
            Y = FLU - FRD + BLD - BRU
            Z = FLU - FRD - BLD + BRU
            btn_loadA["text"] = "Load A format"
        else:
            btn_loadB["text"] = "Loading"
            Wname = filedialog.askopenfilename(title="Load W signal", filetypes=[("WAV Audio", "*.wav")])
            if len(Wname)==0:
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            Xname = filedialog.askopenfilename(title="Load X signal", filetypes=[("WAV Audio", "*.wav")])
            if len(Xname)==0:
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            Yname = filedialog.askopenfilename(title="Load Y signal", filetypes=[("WAV Audio", "*.wav")])
            if len(Yname)==0:
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            Zname = filedialog.askopenfilename(title="Load Z signal", filetypes=[("WAV Audio", "*.wav")])
            if len(Zname)==0:
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            [W,Wfs] = sf.read(Wname)
            [X,Xfs] = sf.read(Xname)
            [Y,Yfs] = sf.read(Yname)
            [Z,Zfs] = sf.read(Zname)
            if len(W)!=len(Y) or len(W)!=len(X) or len(W)!=len(Z):    
                messagebox.showerror("Error", message="Signals length mismatch")
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            if len(np.shape(W))!=1 or len(np.shape(X))!=1 or len(np.shape(Y))!=1 or len(np.shape(Z))!=1:
                messagebox.showerror("Error", message="Signals must have 1 channel")
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            if Wfs!=Xfs or Wfs!=Yfs or Wfs!=Zfs:
                messagebox.showerror("Error", message="Signals sample rate mismatch")
                btn_loadB["text"] = "Load B format"
                showhidecontrols(1)
                return
            if mestype=="sweep":
                InvFiltname= filedialog.askopenfilename(title="Load Inverse Filter", filetypes=[("WAV Audio", "*.wav")])
                if len(InvFiltname)==0:
                    btn_loadB["text"] = "Load B format"
                    showhidecontrols(1)
                    return
                [InvFilt,InvFiltfs] = sf.read(InvFiltname)
                if Wfs!=InvFiltfs:
                    messagebox.showerror("Error", message="Inverse filter sample rate and data sample rate mismatch")    
                    btn_loadB["text"] = "Load B format"
                    showhidecontrols(1)
                    return
                if len(np.shape(InvFilt))!=1:
                    messagebox.showerror("Error", message="Inverse filter must have 1 channel")
                    btn_loadB["text"] = "Load B format"
                    showhidecontrols(1)
                    return
                W = sc.fftconvolve(InvFilt, W, mode="valid")
                X = sc.fftconvolve(InvFilt, X, mode="valid")
                Y = sc.fftconvolve(InvFilt, Y, mode="valid")
                Z = sc.fftconvolve(InvFilt, Z, mode="valid")
            fs = Wfs
            btn_loadB["text"] = "Load B format"
        list.clear(Wsignal)
        list.clear(Ysignal)
        list.clear(Xsignal)
        list.clear(Zsignal)
    
        list.append(Wsignal,W)
        list.append(Xsignal,X)
        list.append(Ysignal,Y)
        list.append(Zsignal,Z)
    except:
        messagebox.showerror("Error", message="Load fail")
    showhidecontrols(1)
    
    return


def func_analyze():
    try: 
        if len(Wsignal)==0:
            messagebox.showerror("Error", message="No data loaded")
            return
        if correction.get()==1:
            try:
                float(entry_center2mic.get())
            except:
                messagebox.showerror("Error", message="Entry center to mic distance")
                return
            if float(entry_center2mic.get())<=0:
                messagebox.showerror("Error", message="Entry center to mic distance")
                return
        
        
        showhidecontrols(0)
        cmbx_pltselect["values"] = "Hedgehog"
        W = Wsignal[0]
        Y = Ysignal[0]
        X = Xsignal[0]
        Z = Zsignal[0]
    
        global data
        global Ix
        global Iy
        global Iz
    
        if correction.get()==1:
            dist = float(entry_center2mic.get())/100
            b1w = 1j*dist/343
            b1x =  (1j/3)*dist
            b2 = (-1/3)*(dist/343)**2
            a1 =  (1j/3)*dist/343
            (zw, pw) = sc.bilinear(np.array([1,b1w,b2]),np.array([1,a1]),fs)
            (zx, px)  = sc.bilinear(np.array([1,b1x,b2]),np.array([1,a1]),fs)
            W = sc.lfilter(zw,pw,W)
            X = sc.lfilter(zx,px,X)*np.sqrt(6)
            Y = sc.lfilter(zx,px,Y)*np.sqrt(6)
            Z= sc.lfilter(zx,px,Z)*np.sqrt(6)
            
        freclim = (2 ** (1 / 2))  
        finf = (2 / fs) * float(cmbx_minband.get()) / freclim
        fsup = (2 / fs) * float(cmbx_maxband.get()) * freclim
        sos = sc.butter(8, [finf,fsup], btype="band", output="sos", analog=False)
        W = sc.sosfiltfilt(sos,W)
        X = sc.sosfiltfilt(sos,X)
        Y = sc.sosfiltfilt(sos,Y)
        Z = sc.sosfiltfilt(sos,Z)
        
        W2 = W**2
        IRstart = int(np.argmax(W2))
        W2 = W2[IRstart:]
        W2[W2==0] = 0.00000000001
        W = W[IRstart:]
        X = X[IRstart:]
        Y = Y[IRstart:]
        Z = Z[IRstart:]
        IRend = metodo_propio(10*np.log10(W2), fs, 4000)
        W = W[:IRend]
        X = X[:IRend]
        Y = Y[:IRend]
        Z = Z[:IRend]
        
        maxlevel = max(max(np.abs(W)), max(np.abs(X)), max(np.abs(Y)), max(np.abs(Z)))    
        W = W/maxlevel
        X = X/maxlevel
        Y = Y/maxlevel
        Z = Z/maxlevel
    
        Ix = W*X
        Iy = W*Y
        Iz = W*Z
        
        samples = int(np.round(fs*float(cmbx_winlen.get())/1000/4)*4)
        over = int(samples*float(cmbx_overlap.get())/100)
        non_over = samples-over
        num_windows = int(np.ceil((len(W)-over)/non_over))
        zpad = int(num_windows*non_over-(len(W)-over))
        np.pad(W, (0,zpad), "constant")
        np.pad(X, (0,zpad), "constant")
        np.pad(Y, (0,zpad), "constant")
        np.pad(Z, (0,zpad), "constant")
        I = np.zeros([num_windows,3])
        vector_director = np.zeros([num_windows,3])
        for k in range(num_windows):
            I[k,0] = np.mean(Ix[k*non_over:samples+k*non_over])
            I[k,1] = np.mean(Iy[k*non_over:samples+k*non_over])
            I[k,2] = np.mean(Iz[k*non_over:samples+k*non_over])        
            vector_director[k] = I[k]/np.sqrt(np.sum(I[k]**2))
            
        magnitud = np.sqrt(np.sum(I**2, axis=1))
        magnitud = magnitud/magnitud[0]
        data = np.zeros([num_windows,4])
        data[:,0] = np.arange(num_windows)*non_over/fs
        data[:,1] = 10*np.log10(magnitud)
        data[:,2] = np.angle(vector_director[:,0]+1j*vector_director[:,1])*180/np.pi
        data[:,3] = np.arccos(vector_director[:,2])*180/np.pi
        
        idx = data[:,1] >= float(cmbx_treshold.get())
        data = data[idx]
        magnitud = 10*np.log10(magnitud[idx])
        magnitud = magnitud-2*min(magnitud)
        magnitud = magnitud/max(magnitud)
        vector_director = vector_director[idx]
        
        
        graphplanxy.cla()
        graphplanxz.cla()
        graphplanyz.cla()
        graphplanxy.axis("off")
        graphplanxz.axis("off")
        graphplanyz.axis("off")
        graph3d.lines = []
        graphxy.lines = []
        graphxz.lines = []
        graphyz.lines = []
        graphw.lines = []
        cmbx_pltselect.current(0)
        changeplot(0)
        scale_rotatexy.set(0)
        scale_rotatexz.set(0)
        scale_rotateyz.set(0)
        scale_zoomxy.set(1)
        scale_zoomxz.set(1)
        scale_zoomyz.set(1)
        chkbtn_invertxy.deselect()
        chkbtn_invertxz.deselect()
        chkbtn_invertyz.deselect()
        if len(fig3d.axes)==2: fig3d.axes[1].remove()
    
    
    
        
        if len(planxy)!=0:
            graphplanxy.imshow(planxy[0], extent=[-1,3,-1,4*np.shape(planxy[0])[0]/np.shape(planxy[0])[1]-1])
            cmbx_pltselect["values"] = list(cmbx_pltselect["values"]) + ["XY Plan"]
        if len(planxz)!=0:
            graphplanxz.imshow(planxz[0], extent=[-1,3,-1,4*np.shape(planxz[0])[0]/np.shape(planxz[0])[1]-1])
            cmbx_pltselect["values"] = list(cmbx_pltselect["values"]) + ["XZ Plan"]
        if len(planyz)!=0:
            graphplanyz.imshow(planyz[0],  extent=[-1,3,-1,4*np.shape(planyz[0])[0]/np.shape(planyz[0])[1]-1])
            cmbx_pltselect["values"] = list(cmbx_pltselect["values"]) + ["YZ Plan"]
        
        
        cjet = plt.cm.get_cmap("jet")
        if len(magnitud)!=1:
            cidx = 1-data[:,0]/data[-1,0]
            fig3d.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,vmax=data[-1,0]*1000),
                                                 cmap=cjet.reversed()), label="Time [ms]", aspect=50)
        
        graph3d.plot([0,vector_director[0,0]], [0,vector_director[0,1]], 
                     [0,vector_director[0,2]], color="k", picker=True, 
                     label=str(data[0,0]*1000)+"ms"+"    "+
                     str(np.round(data[0,1],1))+"dBdirect"+"    "+
                     "azim:"+str(np.round(data[0,2],1))+"°"+"    "+
                     "elev:"+str(np.round(data[0,3],1))+"°")
        graphxy.plot([0,vector_director[0,0]], [0,vector_director[0,1]], color="k")
        graphxz.plot([0,vector_director[0,0]], [0,vector_director[0,2]], color="k")
        graphyz.plot([0,vector_director[0,1]], [0,vector_director[0,2]], color="k")
        graphplanxy.plot([0,vector_director[0,0]], [0,vector_director[0,1]], color="k")
        graphplanxz.plot([0,vector_director[0,0]], [0,vector_director[0,2]], color="k")
        graphplanyz.plot([0,vector_director[0,1]], [0,vector_director[0,2]], color="k")
        graphw.plot(1000*np.arange(int(data[-1,0]*fs)+samples)/fs,W[0:int(data[-1,0]*fs)+samples],"k")
        graphw.set_xlim(xmax=1000*int(data[-1,0]*fs-1+samples)/fs)
        
        for k in range(1,len(magnitud)):
            graph3d.plot([0,vector_director[k,0]*magnitud[k]], 
                         [0,vector_director[k,1]*magnitud[k]], 
                         [0,vector_director[k,2]*magnitud[k]], color=cjet(cidx[k]), 
                         picker=True, label=str(data[k,0]*1000)+"ms"+"    "+
                         str(np.round(data[k,1],1))+"dBdirect"+"    "+
                         "azim:"+str(np.round(data[k,2],1))+"°"+"    "+
                         "elev:"+str(np.round(data[k,3],1))+"°")
            graphxy.plot([0,vector_director[k,0]*magnitud[k]], 
                         [0,vector_director[k,1]*magnitud[k]], color=cjet(cidx[k]))
            graphxz.plot([0,vector_director[k,0]*magnitud[k]], 
                         [0,vector_director[k,2]*magnitud[k]], color=cjet(cidx[k]))
            graphyz.plot([0,vector_director[k,1]*magnitud[k]], 
                         [0,vector_director[k,2]*magnitud[k]], color=cjet(cidx[k]))
            graphplanxy.plot([0,vector_director[k,0]*magnitud[k]], 
                         [0,vector_director[k,1]*magnitud[k]], color=cjet(cidx[k]))
            graphplanxz.plot([0,vector_director[k,0]*magnitud[k]], 
                         [0,vector_director[k,2]*magnitud[k]], color=cjet(cidx[k]))
            graphplanyz.plot([0,vector_director[k,1]*magnitud[k]], 
                         [0,vector_director[k,2]*magnitud[k]], color=cjet(cidx[k]))
        
        canv3d.draw()
        canvw.draw()
        canvxy.draw()
        canvxz.draw()
        canvyz.draw() 
        canvplanxy.draw()
        canvplanxz.draw()
        canvplanyz.draw()
    except:
        messagebox.showerror("Error", message="Process fail")
    showhidecontrols(1)

    return


def showhidecontrols(show_hide):
    if show_hide==1:
        cmbx_maxband["state"] = "readonly"
        cmbx_minband["state"] = "readonly"
        cmbx_overlap["state"] = "readonly"
        cmbx_pltselect["state"] = "readonly"
        cmbx_treshold["state"] = "readonly"
        cmbx_winlen["state"] = "readonly"
        btn_expdata["state"] = NORMAL
        btn_expintensity["state"] = NORMAL
        btn_expplot["state"] = NORMAL
        btn_graph["state"] = NORMAL
        btn_loadplan["state"] = NORMAL
        btn_loadA["state"] = NORMAL
        btn_loadB["state"] = NORMAL
        chkbtn_correction["state"] = NORMAL
        chkbtn_invertxy["state"] = NORMAL
        chkbtn_invertxz["state"] = NORMAL
        chkbtn_invertyz["state"] = NORMAL
        correctionchange()
        
    else:
        cmbx_maxband["state"] = DISABLED
        cmbx_minband["state"] = DISABLED
        cmbx_overlap["state"] = DISABLED
        cmbx_pltselect["state"] = DISABLED
        cmbx_treshold["state"] = DISABLED
        cmbx_winlen["state"] = DISABLED
        entry_center2mic["state"] = DISABLED
        btn_expdata["state"] = DISABLED
        btn_expintensity["state"] = DISABLED
        btn_expplot["state"] = DISABLED
        btn_graph["state"] = DISABLED
        btn_loadplan["state"] = DISABLED
        btn_loadA["state"] = DISABLED
        btn_loadB["state"] = DISABLED
        chkbtn_correction["state"] = DISABLED
        chkbtn_invertxy["state"] = DISABLED
        chkbtn_invertxz["state"] = DISABLED
        chkbtn_invertyz["state"] = DISABLED

    return


def showlegend(event):
    
    graph3d.legend(handles=[event.artist])
    canv3d.draw()
    
    return


def hidelegend(event):
    
    graph3d.legend("")
    graph3d.get_legend().remove()
    canv3d.draw()
    
    return


def metodo_propio(data, fs, k):
    
    mmf = mediamovil(data, k)
    mmf = np.concatenate([mmf, np.ones([fs]) * mmf[-1]])
    indexmax = np.argmax(mmf)
    levelmax = np.max(mmf)
    M = (mmf[-1] - levelmax) / (len(mmf) - indexmax)
    B = (levelmax - M * indexmax)
    cut = np.argmax(M * range(len(mmf))[indexmax:] + B - mmf[indexmax:]) + indexmax
    cut = np.int(cut)
    mmf = np.concatenate([mmf[0:cut], np.ones([fs]) * mmf[cut]])
    M = (mmf[-1] - levelmax) / (len(mmf) - indexmax)
    B = (levelmax - M * indexmax)
    cut = np.argmax(M * range(len(mmf))[indexmax:] + B - mmf[indexmax:]) + indexmax
    cut = np.int(cut)
    
    return cut


def mediamovil(data, k):
    
    w = np.concatenate([np.zeros([len(data) - 1]), np.ones([k]) / k])
    dataz = np.concatenate([data, np.zeros([k - 1])])
    fftdataz = np.fft.rfft(dataz)
    fftw = np.fft.rfft(w)
    mmf = np.fft.irfft(fftdataz * fftw)
    mmf = mmf[0:len(data) - k + 1]
    
    return mmf


def func_limitband(event):
    
    cmbx_maxband["values"] = bands[cmbx_minband.current():]
    cmbx_minband["values"] = bands[0:cmbx_maxband.current() + cmbx_minband.current() + 1]
    
    return


def func_rotate3dazim(*arg):
    
    if arg[0]=="moveto":
        idx = float(arg[1])
        scrollbarH.set(idx,idx)
        graph3d.view_init(graph3d.elev,idx*360)
        canv3d.draw()
    if arg[0]=="scroll":
        idx = float(scrollbarH.get()[0])+int(arg[1])*0.01
        scrollbarH.set(idx,idx)
        graph3d.view_init(graph3d.elev,idx*360)
        canv3d.draw()
        
    return


def func_rotate3delev(*arg):
    
    if arg[0]=="moveto":
        idx = float(arg[1])
        scrollbarV.set(idx,idx)
        graph3d.view_init(-idx*180+90,graph3d.azim)
        canv3d.draw()
    if arg[0]=="scroll":
        idx = float(scrollbarV.get()[0])+int(arg[1])*0.01
        scrollbarV.set(idx,idx)
        graph3d.view_init(-idx*180+90,graph3d.azim)
        canv3d.draw() 
        
    return


def changeplot(event):
    
    if cmbx_pltselect.get()=="Hedgehog":
        figplanxyframe.grid_remove()
        figplanxzframe.grid_remove()
        figplanyzframe.grid_remove()
        fig2dframe.grid()
        fig3dframe.grid()
        figwframe.grid()
        scale_rotatexy.grid_remove()
        scale_zoomxy.grid_remove()
        chkbtn_invertxy.grid_remove()
        scale_rotatexz.grid_remove()
        scale_zoomxz.grid_remove()
        chkbtn_invertxz.grid_remove()
        scale_rotateyz.grid_remove()
        scale_zoomyz.grid_remove()
        chkbtn_invertyz.grid_remove()

    elif cmbx_pltselect.get()=="XY Plan":
        figplanxyframe.grid()
        figplanxzframe.grid_remove()
        figplanyzframe.grid_remove()
        fig2dframe.grid_remove()
        fig3dframe.grid_remove()
        figwframe.grid_remove()
        scale_rotatexy.grid()
        scale_zoomxy.grid()
        chkbtn_invertxy.grid()
        scale_rotatexz.grid_remove()
        scale_zoomxz.grid_remove()
        chkbtn_invertxz.grid_remove()
        scale_rotateyz.grid_remove()
        scale_zoomyz.grid_remove()
        chkbtn_invertyz.grid_remove()
        
    elif cmbx_pltselect.get()=="XZ Plan":
        figplanxyframe.grid_remove()
        figplanxzframe.grid()
        figplanyzframe.grid_remove()
        fig2dframe.grid_remove()
        fig3dframe.grid_remove()
        figwframe.grid_remove()
        scale_rotatexy.grid_remove()
        scale_zoomxy.grid_remove()
        chkbtn_invertxy.grid_remove()
        scale_rotatexz.grid()
        scale_zoomxz.grid()
        chkbtn_invertxz.grid()
        scale_rotateyz.grid_remove()
        scale_zoomyz.grid_remove()
        chkbtn_invertyz.grid_remove()
    
    elif cmbx_pltselect.get()=="YZ Plan":
        figplanxyframe.grid_remove()
        figplanxzframe.grid_remove()
        figplanyzframe.grid()
        fig2dframe.grid_remove()
        fig3dframe.grid_remove()
        figwframe.grid_remove()
        scale_rotatexy.grid_remove()
        scale_zoomxy.grid_remove()
        chkbtn_invertxy.grid_remove()
        scale_rotatexz.grid_remove()
        scale_zoomxz.grid_remove()
        chkbtn_invertxz.grid_remove()
        scale_rotateyz.grid()
        scale_zoomyz.grid()
        chkbtn_invertyz.grid()
        
    return


def correctionchange():
    
    if correction.get()==1:
        lbl_center2mic["state"] = NORMAL
        lbl_center2micunit["state"] = NORMAL
        entry_center2mic["state"] = NORMAL
    else:
        lbl_center2mic["state"] = DISABLED
        lbl_center2micunit["state"] = DISABLED
        entry_center2mic["state"] = DISABLED
    
    return


def func_expdata():
    # Export the vectors data to .Csv
    if len(graphxy.lines)==0:  
        messagebox.showerror("Error", message="No data analysed")
        return   
    filename = filedialog.asksaveasfilename(title="Export vectors data", filetypes=[("CSV File", "*.csv")],
                                                defaultextension=[("CSV File", "*.csv")])
    if not filename: return
    file = open(filename, mode='w')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Time[s]","Level[dB]","Azimuth[°]","Elevation[°]"])
    for k in range(len(data[:,0])):
        writer.writerow(list(data[k,:]))
    file.close()
    return


def func_expintensity():
    # Export the instantaneous intensity to .csv
    if len(graphxy.lines)==0:  
        messagebox.showerror("Error", message="No data analysed")
        return   
    filename = filedialog.asksaveasfilename(title="Export instantaneous intensity", filetypes=[("CSV File", "*.csv")],
                                                defaultextension=[("CSV File", "*.csv")])
    if not filename: return
    file = open(filename, mode='w')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Ix","Iy","Iz"])
    for k in range(len(Ix)):
        writer.writerow([Ix[k],Iy[k],Iz[k]])
    file.close()
    return


def func_expplot():
    
    if cmbx_pltselect.get()=="Hedgehog":
        filename = filedialog.asksaveasfile(title="Save 3D hedgehog plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: fig3d.savefig(filename.name, transparent=True, dpi="figure")
        filename = filedialog.asksaveasfile(title="Save XY hedgehog plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figxy.savefig(filename.name, transparent=True, dpi="figure")
        filename = filedialog.asksaveasfile(title="Save XZ hedgehog plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figxz.savefig(filename.name, transparent=True, dpi="figure")
        filename = filedialog.asksaveasfile(title="Save YZ hedgehog plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figyz.savefig(filename.name, transparent=True, dpi="figure")
        filename = filedialog.asksaveasfile(title="Save Wsignal plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figw.savefig(filename.name, transparent=True, dpi="figure")
    elif cmbx_pltselect.get()=="XY Plan":
        filename = filedialog.asksaveasfile(title="Save XY plan plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figplanxy.savefig(filename.name, transparent=True, dpi="figure")
    elif cmbx_pltselect.get()=="XZ Plan":
        filename = filedialog.asksaveasfile(title="Save XZ plan plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figplanxz.savefig(filename.name, transparent=True, dpi="figure")
    elif cmbx_pltselect.get()=="YZ Plan":
        filename = filedialog.asksaveasfile(title="Save YZ plan plot", filetypes=[("PNG Image", "*.png")],
                                            defaultextension=[("PNG Image", "*.png")])
        if filename: figplanyz.savefig(filename.name, transparent=True, dpi="figure")
    
    return


def func_aspecthedgehogxy(*kwargs):
    
    a = np.cos(float(scale_rotatexy.get())*np.pi/180)*float(scale_zoomxy.get())
    b = np.sin(float(scale_rotatexy.get())*np.pi/180)*float(scale_zoomxy.get())
    for k in range(len(graphplanxy.lines)):
        xv = graphxy.lines[k].get_xdata()[1]
        yv = graphxy.lines[k].get_ydata()[1]
        xv2 = xv*a-yv*b
        yv2 =  yv*a+xv*b
        if invertxy.get()==1: yv2=-yv2
        graphplanxy.lines[k].set_xdata([graphplanxy.lines[k].get_xdata()[0], xv2+graphplanxy.lines[k].get_xdata()[0]])
        graphplanxy.lines[k].set_ydata([graphplanxy.lines[k].get_ydata()[0], yv2+graphplanxy.lines[k].get_ydata()[0]])
    canvplanxy.draw()
    
    return


def func_aspecthedgehogxz(*kwargs):
    
    a = np.cos(float(scale_rotatexz.get())*np.pi/180)*float(scale_zoomxz.get())
    b = np.sin(float(scale_rotatexz.get())*np.pi/180)*float(scale_zoomxz.get())
    for k in range(len(graphplanxz.lines)):
        xv = graphxz.lines[k].get_xdata()[1]
        yv = graphxz.lines[k].get_ydata()[1]
        xv2 = xv*a-yv*b
        yv2 =  yv*a+xv*b
        if invertxz.get()==1: yv2=-yv2
        graphplanxz.lines[k].set_xdata([graphplanxz.lines[k].get_xdata()[0], xv2+graphplanxz.lines[k].get_xdata()[0]])
        graphplanxz.lines[k].set_ydata([graphplanxz.lines[k].get_ydata()[0], yv2+graphplanxz.lines[k].get_ydata()[0]])
    canvplanxz.draw()
    
    return


def func_aspecthedgehogyz(*kwargs):
    
    a = np.cos(float(scale_rotateyz.get())*np.pi/180)*float(scale_zoomyz.get())
    b = np.sin(float(scale_rotateyz.get())*np.pi/180)*float(scale_zoomyz.get())
    for k in range(len(graphplanyz.lines)):
        xv = graphyz.lines[k].get_xdata()[1]
        yv = graphyz.lines[k].get_ydata()[1]
        xv2 = xv*a-yv*b
        yv2 =  yv*a+xv*b
        if invertyz.get()==1: yv2=-yv2
        graphplanyz.lines[k].set_xdata([graphplanyz.lines[k].get_xdata()[0], xv2+graphplanyz.lines[k].get_xdata()[0]])
        graphplanyz.lines[k].set_ydata([graphplanyz.lines[k].get_ydata()[0], yv2+graphplanyz.lines[k].get_ydata()[0]])
    canvplanyz.draw()
    
    return


def placehedgehogxy(event):
    
    if event.inaxes==None: return
    origenx = -graphplanxy.lines[0].get_xdata()[0]+event.xdata
    origeny = -graphplanxy.lines[0].get_ydata()[0]+event.ydata
    for k in range(len(graphplanxy.lines)):
        graphplanxy.lines[k].set_xdata(graphplanxy.lines[k].get_xdata()+origenx)
        graphplanxy.lines[k].set_ydata(graphplanxy.lines[k].get_ydata()+origeny)
    canvplanxy.draw()
    
    return


def placehedgehogxz(event):
    
    if event.inaxes==None: return
    origenx = -graphplanxz.lines[0].get_xdata()[0]+event.xdata
    origeny = -graphplanxz.lines[0].get_ydata()[0]+event.ydata
    for k in range(len(graphplanxy.lines)):
        graphplanxz.lines[k].set_xdata(graphplanxz.lines[k].get_xdata()+origenx)
        graphplanxz.lines[k].set_ydata(graphplanxz.lines[k].get_ydata()+origeny)
    canvplanxz.draw()
    
    return


def placehedgehogyz(event):
    
    if event.inaxes==None: return
    origenx = -graphplanyz.lines[0].get_xdata()[0]+event.xdata
    origeny = -graphplanyz.lines[0].get_ydata()[0]+event.ydata
    for k in range(len(graphplanyz.lines)):
        graphplanyz.lines[k].set_xdata(graphplanyz.lines[k].get_xdata()+origenx)
        graphplanyz.lines[k].set_ydata(graphplanyz.lines[k].get_ydata()+origeny)
    canvplanyz.draw()
    
    return


warnings.filterwarnings("ignore", category=UserWarning)

planxy = []
planxz = []
planyz = []
Wsignal = []
Xsignal = []
Ysignal = []
Zsignal = []
bands = ["31.5", "63", "125", "250", "500", "1000", "2000", "4000", "8000", "16000"]

rir3d = Tk()
rir3d.title("Brojo Software - Room impulse response 3D spatial analyzer for 1st order Ambisonic measurment")
rir3d.rowconfigure(0, weight=1)
rir3d.rowconfigure(1, weight=1)
rir3d.columnconfigure(1, weight=1)
rir3d.columnconfigure(2, weight=1)
rir3d.state("zoomed")
r, g, b = rir3d.winfo_rgb(rir3d["bg"])

menu_frame = Frame(rir3d)
menu_frame.grid(row=0, column=0, rowspan=2, padx=3, pady=3, sticky="ewn")
menu_frame.columnconfigure(1,weight=1)

btn_loadA = Button(menu_frame, text="Load A format", command=lambda:func_load("A"))
btn_loadA.grid(row=0, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

btn_loadB = Button(menu_frame, text="Load B format", command=lambda:func_load("B"))
btn_loadB.grid(row=1, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

btn_loadplan = Button(menu_frame, text="Load plan view", command=func_loadplan)
btn_loadplan.grid(row=2, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

lbl_settings = Label(menu_frame, text="Settings")
lbl_settings.grid(row=3, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

lbl_minband = Label(menu_frame, text="Band min:")
lbl_minband.grid(row=4, column=0, padx=3, sticky="ens")

cmbx_minband = Combobox(menu_frame, width=3, values=bands, state="readonly")
cmbx_minband.current(0)
cmbx_minband.bind("<<ComboboxSelected>>", func_limitband)
cmbx_minband.grid(row=4, column=1, pady=2, sticky="ewns")

lbl_minbandunit = Label(menu_frame, text="Hz")
lbl_minbandunit.grid(row=4, column=2, padx=3, sticky="wns")

lbl_maxband = Label(menu_frame, text="Band max:")
lbl_maxband.grid(row=5, column=0, padx=3, sticky="ens")

cmbx_maxband = Combobox(menu_frame, width=3, values=bands, state="readonly")
cmbx_maxband.current(len(bands)-1)
cmbx_maxband.bind("<<ComboboxSelected>>", func_limitband)
cmbx_maxband.grid(row=5, column=1, pady=2, sticky="ewns")

lbl_maxbandunit = Label(menu_frame, text="Hz")
lbl_maxbandunit.grid(row=5, column=2, padx=3, sticky="wns")

lbl_winlen = Label(menu_frame, text="Win. length:")
lbl_winlen.grid(row=6, column=0, padx=3, sticky="ens")

lbl_winlenunit = Label(menu_frame, text="ms")
lbl_winlenunit.grid(row=6, column=2, padx=3, sticky="wns")

cmbx_winlen = Combobox(menu_frame, values=(1,2,5,10), width=3, state="readonly")
cmbx_winlen.grid(row=6, column=1, pady=2, sticky="ewns")
cmbx_winlen.current(0)

lbl_overlap = Label(menu_frame, text="Overlap:")
lbl_overlap.grid(row=7, column=0, padx=3, sticky="ens")

cmbx_overlap = Combobox(menu_frame, values=(0,25,50,75), width=3, state="readonly")
cmbx_overlap.grid(row=7, column=1, pady=2, sticky="ewns")
cmbx_overlap.current(1)

lbl_overlapunit = Label(menu_frame, text="%")
lbl_overlapunit.grid(row=7, column=2, padx=3, sticky="wns")

lbl_treshold = Label(menu_frame, text="Treshold:")
lbl_treshold.grid(row=8, column=0, padx=3, sticky="ens")

cmbx_treshold = Combobox(menu_frame, values=(-5,-10,-15,-20,-25,-30,-35,-40,-45,-50,-55,-60), width=3, state="readonly")
cmbx_treshold.grid(row=8, column=1, pady=2, sticky="ewns")
cmbx_treshold.current(7)

lbl_tresholdunit = Label(menu_frame, text="dB")
lbl_tresholdunit.grid(row=8, column=2, padx=3, sticky="wns")

correction = IntVar()
chkbtn_correction = Checkbutton(menu_frame, text="Non-coincidence correction", variable=correction, command=correctionchange)
chkbtn_correction.grid(row=9, column=0, columnspan=3, padx=3, pady=2, sticky="ewns")
chkbtn_correction.select()

lbl_center2mic = Label(menu_frame, text="Center to mic:")
lbl_center2mic.grid(row=10, column=0, padx=3, sticky="ens")

entry_center2mic = Entry(menu_frame, width=3)
entry_center2mic.grid(row=10, column=1, pady=2, sticky="ewns")

lbl_center2micunit = Label(menu_frame, text="cm")
lbl_center2micunit.grid(row=10, column=2, padx=3, sticky="wns")

btn_graph = Button(menu_frame, text="Analyze", font=("Arial", 11, "bold"), command=func_analyze)
btn_graph.grid(row=11, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

btn_expdata = Button(menu_frame, text="Export vectors data", command=func_expdata)
btn_expdata.grid(row=12, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

btn_expintensity = Button(menu_frame, text="Export inst. intensity", command=func_expintensity)
btn_expintensity.grid(row=13, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

btn_expplot = Button(menu_frame, text="Export current plot", command=func_expplot)
btn_expplot.grid(row=14, column=0, columnspan=3, padx=3, pady=3, sticky="ewns")

lbl_pltselect = Label(menu_frame, text="Plot:")
lbl_pltselect.grid(row=15, column=0, padx=2, sticky="ens")

cmbx_pltselect = Combobox(menu_frame, values=("Hedgehog"), state="readonly", width=4)
cmbx_pltselect.grid(row=15, column=1, columnspan=2, pady=3, sticky="ewns")
cmbx_pltselect.current(0)
cmbx_pltselect.bind("<<ComboboxSelected>>", changeplot)

scale_rotatexy = Scale(menu_frame, label="Rotate hedgehog", from_=0, to=359, resolution=-1, orient=HORIZONTAL, command=func_aspecthedgehogxy)
scale_rotatexy.grid(row=16, column=0, columnspan=3, pady=2, sticky="ewns")
scale_rotatexy.grid_remove()

scale_zoomxy = Scale(menu_frame, label="Scale hedgehog", from_=0, to=2, resolution=-1, orient=HORIZONTAL, command=func_aspecthedgehogxy)
scale_zoomxy.grid(row=17, column=0, columnspan=3, pady=2, sticky="ewns")
scale_zoomxy.set(1)
scale_zoomxy.grid_remove()

invertxy = IntVar()
chkbtn_invertxy = Checkbutton(menu_frame, text="Invert hedgehog", variable=invertxy, command=func_aspecthedgehogxy)
chkbtn_invertxy.grid(row=18, column=0, columnspan=3, padx=3, pady=2, sticky="ewns")
chkbtn_invertxy.grid_remove()

scale_rotatexz = Scale(menu_frame, label="Rotate hedgehog", from_=0, to=359, resolution=-1, orient=HORIZONTAL, command=func_aspecthedgehogxz)
scale_rotatexz.grid(row=16, column=0, columnspan=3, pady=2, sticky="ewns")
scale_rotatexz.grid_remove()

scale_zoomxz = Scale(menu_frame, label="Scale hedgehog", from_=0, to=2, resolution=-1, orient=HORIZONTAL, command=func_aspecthedgehogxz)
scale_zoomxz.grid(row=17, column=0, columnspan=3, pady=2, sticky="ewns")
scale_zoomxz.set(1)
scale_zoomxz.grid_remove()

invertxz = IntVar()
chkbtn_invertxz = Checkbutton(menu_frame, text="Invert hedgehog", variable=invertxz, command=func_aspecthedgehogxz)
chkbtn_invertxz.grid(row=18, column=0, columnspan=3, padx=3, pady=2, sticky="ewns")
chkbtn_invertxz.grid_remove()

scale_rotateyz = Scale(menu_frame, label="Rotate hedgehog", from_=0, to=359, resolution=-1, orient=HORIZONTAL, command=func_aspecthedgehogyz)
scale_rotateyz.grid(row=16, column=0, columnspan=3, pady=2, sticky="ewns")
scale_rotateyz.grid_remove()

scale_zoomyz = Scale(menu_frame, label="Scale hedgehog", from_=0, to=2, resolution=-1, orient=HORIZONTAL, command=func_aspecthedgehogyz)
scale_zoomyz.grid(row=17, column=0, columnspan=3, pady=2, sticky="ewns")
scale_zoomyz.set(1)
scale_zoomyz.grid_remove()

invertyz = IntVar()
chkbtn_invertyz = Checkbutton(menu_frame, text="Invert hedgehog", variable=invertyz, command=func_aspecthedgehogyz)
chkbtn_invertyz.grid(row=18, column=0, columnspan=3, padx=3, pady=2, sticky="ewns")
chkbtn_invertyz.grid_remove()

fig2dframe = Frame(rir3d)
fig2dframe.grid(row=0, column=2, rowspan=2, sticky="ewns")
fig2dframe.rowconfigure(0,weight=1)
fig2dframe.rowconfigure(1,weight=1)
fig2dframe.rowconfigure(2,weight=1)
fig2dframe.columnconfigure(0,weight=1)

figxy = Figure(figsize=(1,1), constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphxy = figxy.add_subplot(111)
graphxy.set_xlabel("X",labelpad=0)
graphxy.set_xticks([-1,1])
graphxy.set_xticklabels(["Back","Front"])
graphxy.set_xlim([-1,1])
graphxy.set_ylabel("Y",labelpad=0)
graphxy.set_yticks([-1,1])
graphxy.set_yticklabels(["Right","Left"])
graphxy.set_ylim([-1,1])
graphxy.grid(False)
graphxy.set_box_aspect(1)
canvxy = FigureCanvasTkAgg(figxy, master=fig2dframe)
canvxy.draw()
get_widzxy = canvxy.get_tk_widget()
get_widzxy.grid(row=0, column=0, sticky="ewns")

figxz = Figure(figsize=(1,1), constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphxz = figxz.add_subplot(111)
graphxz.set_xlabel("X",labelpad=0)
graphxz.set_xticks([-1,1])
graphxz.set_xticklabels(["Back","Front"])
graphxz.set_xlim([-1,1])
graphxz.set_ylabel("Z",labelpad=0)
graphxz.set_yticks([-1,1])
graphxz.set_yticklabels(["Down","Up"])
graphxz.set_ylim([-1,1])
graphxz.grid(False)
graphxz.set_box_aspect(1)
canvxz = FigureCanvasTkAgg(figxz, master=fig2dframe)
canvxz.draw()
get_widzxz = canvxz.get_tk_widget()
get_widzxz.grid(row=1, column=0, sticky="ewns")

figyz = Figure(figsize=(1,1), constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphyz = figyz.add_subplot(111)
graphyz.set_xlabel("Y",labelpad=0)
graphyz.set_xticks([-1,1])
graphyz.set_xticklabels(["Right","Left"])
graphyz.set_xlim([-1,1])
graphyz.set_ylabel("Z",labelpad=0)
graphyz.set_yticks([-1,1])
graphyz.set_yticklabels(["Down","Up"])
graphyz.set_ylim([-1,1])
graphyz.grid(False)
graphyz.set_box_aspect(1)
canvyz = FigureCanvasTkAgg(figyz, master=fig2dframe)
canvyz.draw()
get_widzyz = canvyz.get_tk_widget()
get_widzyz.grid(row=2, column=0, sticky="ewns")

fig3dframe = Frame(rir3d)
fig3dframe.grid(row=0, column=1, sticky="ewns")
fig3dframe.rowconfigure(0,weight=1)
fig3dframe.columnconfigure(1,weight=1)

fig3d = Figure(constrained_layout=True, figsize=(6,8))
graph3d = fig3d.add_subplot(111, projection="3d")
graph3d.set_xlabel("X")
graph3d.set_xticks([-1,1])
graph3d.set_xticklabels(["Back","Front"])
graph3d.set_xlim([-1,1])
graph3d.set_ylabel("Y",labelpad=0)
graph3d.set_yticks([-1,1])
graph3d.set_yticklabels(["Right","Left"])
graph3d.set_ylim([-1,1])
graph3d.set_zlabel("Z",labelpad=0)
graph3d.set_zticks([-1,1])
graph3d.set_zticklabels(["Down","Up"])
graph3d.set_zlim([-1,1])
graph3d.grid(False)
canv3d = FigureCanvasTkAgg(fig3d, master=fig3dframe)
canv3d.draw()
get_widz3d = canv3d.get_tk_widget()
get_widz3d.grid(row=0, column=1, sticky="ewns")
scrollbarV = Scrollbar(fig3dframe, orient="vertical", command=func_rotate3delev)
scrollbarV.set(1/3,1/3)
scrollbarV.grid(row=0, column=0, sticky="ewns")
scrollbarH = Scrollbar(fig3dframe, orient="horizontal", command=func_rotate3dazim)
scrollbarH.grid(row=1, column=1, sticky="ewns")
scrollbarH.set(5/6,5/6)
canv3d.mpl_connect("pick_event",showlegend)
canv3d.mpl_connect("figure_leave_event",hidelegend)

figwframe = Frame(rir3d)
figwframe.grid(row=1, column=1, sticky="ewns")
figwframe.rowconfigure(0,weight=1)
figwframe.columnconfigure(0,weight=1)

figw = Figure(constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphw = figw.add_subplot(111)
graphw.set_xlabel("Time [ms]",labelpad=0)
graphw.set_ylabel("Amplitude W",labelpad=0)
graphw.grid()
canvw = FigureCanvasTkAgg(figw, master=figwframe)
canvw.draw()
get_widzw = canvw.get_tk_widget()
get_widzw.grid(row=0, column=0, sticky="ewns")

figplanxzframe = Frame(rir3d)
figplanxzframe.grid(row=0, column=1, rowspan=2, columnspan=2, sticky="ewns")
figplanxzframe.grid_remove()
figplanxzframe.rowconfigure(0,weight=1)
figplanxzframe.columnconfigure(0,weight=1)

figplanxz = Figure(constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphplanxz = figplanxz.add_subplot(111)
canvplanxz = FigureCanvasTkAgg(figplanxz, master=figplanxzframe)
canvplanxz.draw()
get_widzplanxz = canvplanxz.get_tk_widget()
get_widzplanxz.grid(row=0, column=0, sticky="ewns")
canvplanxz.mpl_connect("button_press_event", placehedgehogxz)

figplanyzframe = Frame(rir3d)
figplanyzframe.grid(row=0, column=1, rowspan=2, columnspan=2, sticky="ewns")
figplanyzframe.grid_remove()
figplanyzframe.rowconfigure(0,weight=1)
figplanyzframe.columnconfigure(0,weight=1)

figplanyz = Figure(constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphplanyz = figplanyz.add_subplot(111)
canvplanyz = FigureCanvasTkAgg(figplanyz, master=figplanyzframe)
canvplanyz.draw()
get_widzplanyz = canvplanyz.get_tk_widget()
get_widzplanyz.grid(row=0, column=0, sticky="ewns")
canvplanyz.mpl_connect("button_press_event", placehedgehogyz)

figplanxyframe = Frame(rir3d)
figplanxyframe.grid(row=0, column=1, rowspan=2, columnspan=2, sticky="ewns")
figplanxyframe.grid_remove()
figplanxyframe.rowconfigure(0,weight=1)
figplanxyframe.columnconfigure(0,weight=1)

figplanxy = Figure(constrained_layout=True, facecolor=[r / 65536, g / 65536, b / 65536])
graphplanxy = figplanxy.add_subplot(111)
canvplanxy = FigureCanvasTkAgg(figplanxy, master=figplanxyframe)
canvplanxy.draw()
get_widzplanxy = canvplanxy.get_tk_widget()
get_widzplanxy.grid(row=0, column=0, sticky="ewns")
canvplanxy.mpl_connect("button_press_event", placehedgehogxy)


rir3d.mainloop()