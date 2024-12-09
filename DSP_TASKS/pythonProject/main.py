import glob
import math 
import numpy as np
import matplotlib.pyplot as plt
from tkinter import*
from tkinter import ttk
from tkinter import filedialog, messagebox

import ConvTest
import Test1
import Test2
import signalcompare
import comparesignal2
import DerivativeSignal
import Shift_Fold_Signal
import CompareSignal

#Task1
def read_signal_from_file(file_path):
    # Read signal samples from a text file
    with open(file_path) as file:
        listOfLines=file.readlines()
        SignalType=listOfLines[0]
        IsPeriodic=listOfLines[1]
        sampleSize=listOfLines[2]
        del listOfLines[0]
        del listOfLines[0]
        del listOfLines[0]
        samplesX = []
        samplesY = []
        for line in listOfLines:
            if "," in line:
                if "f" in line:
                    splitList = line.strip().split(",")
                    splitList[0]=splitList[0].replace("f","")
                    splitList[1]=splitList[1].replace("f","")
                else:
                    splitList = line.strip().split(",")
            else:
                if "f" in line:
                    splitList = line.strip().split()
                    splitList[0]=splitList[0].replace("f","")
                    splitList[1]=splitList[1].replace("f","")
                else:
                    splitList = line.strip().split()
            sampleX = int(splitList[0])
            sampleY = int(splitList[1])
            samplesX.append(sampleX)
            samplesY.append(sampleY)
    return samplesX, samplesY

def generate_wave_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y = read_signal_from_file(file_path)
    plot_signal_from_file(x, y)

def plot_signal_from_file(samplesX, samplesY):
    two_subplot_fig = plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.plot(samplesX, samplesY, color="orange")
    plt.title("Continuous Signal")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.stem(samplesX, samplesY)
    plt.title("Discrete Signal")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)
    two_subplot_fig.show()

def generate_sinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration):
    # Generate a sinusoidal signal
    if sampling_frequency != 0:
        t = np.arange(0, duration, 1 / sampling_frequency)
    else:
        t = np.arange(0, duration, 1 / analog_frequency)
    signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    return signal

def generate_cosinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration):
    # Generate a cosinusoidal signal
    if sampling_frequency != 0:
        t = np.arange(0, duration, 1 / sampling_frequency)
    else:
        t = np.arange(0, duration, 1 / analog_frequency)
    signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return signal

def plot_signal_wave(samples, title, flag1, flag2=""):

    two_subplot_fig = plt.figure(figsize=(6, 6))
    if flag1 == "error" or flag2 == "error":
        plt.subplot(211)
        plt.plot(samples, color="orange")
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        plt.stem([0, 0, 0, 0, 0, 0, 0, 0])
        plt.title(title)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)
        two_subplot_fig.show()
    else:
        plt.subplot(211)
        plt.plot(samples, color="orange")
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        plt.stem(samples)
        plt.title(title)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)
        two_subplot_fig.show()

def signal_generation_menu_gui():
    # Create a new window for the signal generation menu
    generation_window = Toplevel(root)
    generation_window.title("Signal Generation Menu")
    generation_window.geometry("300x500")

    signal_type_label = Label(generation_window, text="Signal Type(sin || cos):")
    signal_type_label.grid(row=0, column=0)
    signal_type_entry = Entry(generation_window)
    signal_type_entry.grid(row=0, column=1)

    amplitude_label = Label(generation_window, text="Amplitude:")
    amplitude_label.grid(row=1, column=0)
    amplitude_entry = Entry(generation_window)
    amplitude_entry.grid(row=1, column=1)

    phase_shift_label = Label(generation_window, text="Phase Shift (radians):")
    phase_shift_label.grid(row=2, column=0)
    phase_shift_entry = Entry(generation_window)
    phase_shift_entry.grid(row=2, column=1)

    analog_frequency_label = Label(generation_window, text="Analog Frequency (Hz):")
    analog_frequency_label.grid(row=3, column=0)
    analog_frequency_entry = Entry(generation_window)
    analog_frequency_entry.grid(row=3, column=1)

    sampling_frequency_label = Label(generation_window, text="Sampling Frequency (Hz):")
    sampling_frequency_label.grid(row=4, column=0)
    sampling_frequency_entry = Entry(generation_window)
    sampling_frequency_entry.grid(row=4, column=1)

    duration_label = Label(generation_window, text="Duration (seconds):")
    duration_label.grid(row=5, column=0)
    duration_entry = Entry(generation_window)
    duration_entry.grid(row=5, column=1)

    def generate_wave():
        signal_type = signal_type_entry.get()
        amplitude = float(amplitude_entry.get())
        phase_shift = float(phase_shift_entry.get())
        analog_frequency = float(analog_frequency_entry.get())
        sampling_frequency = float(sampling_frequency_entry.get())
        duration = float(duration_entry.get())
        flag2 = ""
        if sampling_frequency == 0:
            flag1 = "error"
        else:
            flag1 = "ok"
            if analog_frequency / sampling_frequency > 0.5 or analog_frequency / sampling_frequency < -0.5:
                flag2 = "error"
            else:
                flag2 = "ok"

        if signal_type == "sin":
            signal1 = generate_sinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration)
            title = f"Sine Wave: A={amplitude}, θ={phase_shift}, f={analog_frequency}, fs={sampling_frequency}"
            plot_signal_wave(signal1, title, flag1, flag2)
        else:
            signal2 = generate_cosinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration)
            title = f"Cosine Wave: A={amplitude}, θ={phase_shift}, f={analog_frequency}, fs={sampling_frequency}"
            plot_signal_wave(signal2, title, flag1, flag2)

    generate_signal_button = Button(generation_window, text="Generate Signal Wave", bg="black", fg="white", command=generate_wave)
    generate_signal_button.grid(row=6, column=1)

#Task2
def OperationsOnSignals():
    Operations_window=Toplevel(root)
    Operations_window.title("Operations On Signals")
    Operations_leble=Label(Operations_window,text="Choose Type of operation")
    Operations_leble.grid(row=0,column=0)
    Operations_ComboBox=ttk.Combobox(Operations_window,values=("Addition","Subtraction","Multiplication"
                                  ,"Squaring Signal","Shifting","Normalize Signal","Accumulation"),state="raeadonly")
    Operations_ComboBox.grid(row=0,column=1)
    def Operations():
        typ=Operations_ComboBox.get()
        if typ=="Addition":
            Addition_Signals()
        elif typ=="Subtraction":
            Subtraction_Signals()
        elif typ=="Multiplication":
            Multiplication_Signals()
        elif typ=="Squaring Signal":
            Squaring_Signals()
        elif typ=="Shifting":
            Shifting_Signals()
        elif typ=="Normalize Signal":
            Normalization_Signals()
        elif typ=="Accumulation":
            Accumulation_Signals()

    def Addition_Signals():
        add_window=Toplevel(Operations_window)
        add_window.title("Addition")
        add_leble=Label(add_window,text="Number of Signals")
        add_leble.grid(row=0,column=0)
        add_entry=Entry(add_window)
        add_entry.grid(row=0,column=1)
        def Addition():
            nsignal=int(add_entry.get())
            list_y=[]
            for n in range(nsignal) :
                Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
                x, y = read_signal_from_file(Fpath)
                list_y.append(y)
            listOut=[sum(Yval)for Yval in zip(*list_y)]
            plt.figure()
            plt.plot(x,listOut)
            plt.title("Additional Signal")
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        add_button=Button(add_window,text="Add",bg="pink",fg="blue",command=Addition)
        add_button.grid(row=2,column=1)

    def Subtraction_Signals():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x1, y1 = read_signal_from_file(Fpath)
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x2, y2 = read_signal_from_file(Fpath)
        listOut=[np.subtract(s1,s2) for s1,s2 in zip(y1,y2)]
        plt.figure()
        plt.plot(x1,listOut)
        plt.title("Subtracted Signal")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def Multiplication_Signals():
        Multiply_window=Toplevel(Operations_window)
        Multiply_window.title("Multiplication")
        Multiply_leble=Label(Multiply_window,text="Enter const to multiply signal")
        Multiply_leble.grid(row=0,column=0)
        Multiply_entry=Entry(Multiply_window)
        Multiply_entry.grid(row=0,column=1)
        def Multiplication():
            const=int(Multiply_entry.get())
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(Fpath)
            #list_y.append(y)
            listOut=[const * Yval for Yval in y]
            plt.figure()
            plt.plot(x,listOut)
            plt.title("Multiplied Signal")
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        Multiply_button=Button(Multiply_window,text="Multiply",bg="pink",fg="blue",command=Multiplication)
        Multiply_button.grid(row=2,column=1)

    def	Squaring_Signals():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(Fpath)
        listOut=[ np.power(s,2) for s in y]
        plt.figure()
        plt.plot(x,listOut)
        plt.title("Squared Signal")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def	Shifting_Signals():
        Shift_window=Toplevel(Operations_window)
        Shift_window.title("Shifting")
        Shift_leble=Label(Shift_window,text="Enter Number to Shift the Signal")
        Shift_leble.grid(row=0,column=0)
        Shift_entry=Entry(Shift_window)
        Shift_entry.grid(row=0,column=1)
        def Shift():
            num=int(Shift_entry.get())
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(Fpath)
            listOut=[s-num for s in x]

            plt.figure()
            plt.plot(listOut,y)
            plt.title("Shifted Signal")
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        Shift_button=Button(Shift_window,text="shift",bg="pink",fg="blue",command=Shift)
        Shift_button.grid(row=2,column=1)

    def Normalization_Signals():
        Normalize_window=Toplevel(Operations_window)
        Normalize_window.title("Normalization")
        Normalize_Label=Label(Normalize_window,text="Enter Normalization Type (-1 to 1) or (0 to 1)")
        Normalize_Label.grid(row=0,column=0)
        Normalize_Entry=Entry(Normalize_window)
        Normalize_Entry.grid(row=0,column=1)
        listOut=[]
        def Normalization():
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(file_path)
            Min_valu = min(y)
            Max_val = max(y)
            normalizeType = Normalize_Entry.get()
            for i in y:
                if normalizeType == "0 to 1":
                    normalized_signal = (i - Min_valu) / (Max_val - Min_valu)
                    listOut.append(normalized_signal)
                elif normalizeType == "-1 to 1":
                    normalized_signal = 2 * ((i - Min_valu) / (Max_val - Min_valu)) - 1
                    listOut.append(normalized_signal)
                else:
                    raise ValueError("Invalid normalize_type.\nEnter -1 to 1 or 0 to 1")
            plt.figure()
            plt.plot(x,listOut)
            plt.title("normalized Signal")
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        normalize_button=Button(Normalize_window,text="normalize",bg="pink",fg="blue",command=Normalization)
        normalize_button.grid(row=2,column=1)

    def Accumulation_Signals():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(Fpath)
        s=0
        listOut=[]
        for i in y:
            s=s+i
            listOut.append(s)
        plt.figure()
        plt.plot(x,listOut)
        plt.title("Accumulated Signal")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    Operations_button=Button(Operations_window,text="Enter",bg="pink",fg="blue",command=Operations)
    Operations_button.grid(row=3,column=1)

#Task3
def Quantization_Signals():
    Quantize_window=Toplevel(root)
    Quantize_window.title("Quantization")
    Quantize_leble=Label(Quantize_window,text="Choose Type Levels or Bits")
    Quantize_leble.grid(row=0,column=0)
    Quantize_ComboBox=ttk.Combobox(Quantize_window,values=("Levels","Bits"),state="raeadonly")
    Quantize_ComboBox.grid(row=0,column=1)
    Quantize_leble2=Label(Quantize_window,text="Enter Number of Levels or Bits")
    Quantize_leble2.grid(row=1,column=0)
    Quantize_entry=Entry(Quantize_window)
    Quantize_entry.grid(row=1,column=1)

    def Quantize():
        typ=Quantize_ComboBox.get()
        if typ == "Bits":
            BitsNum = int(Quantize_entry.get())
            LevelsNum=np.power(2,int(BitsNum))

        else:
            LevelsNum=int(Quantize_entry.get())
            BitsNum= round(math.log2(LevelsNum))
            #Test2.QuantizationTest2()
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(Fpath)
        #Step1
        MaxV=max(y)
        MinV=min(y)
        #Step2
        Delta=(MaxV-MinV)/LevelsNum
        #Step3,4
        InterVals=[]
        z=round(MinV,3)
        MidPoint=[]
        for x in range(LevelsNum):
           p=round((z+Delta),3)
           InterVals.append([z,p])
           MidPoint.append(round((z+z+Delta)/2,3))
           z=p#round(z+Delta,3)
        #Step5
        Val_Index=[]
        Val_Mid=[]
        Encoded=[]
        Error_Quantize=[]
        def Index_and_Mid(y):
            for j in range(len(InterVals)) :
                if y<=InterVals[j][1] and y>=InterVals[j][0]:
                    Val_Index.append(j+1)
                    Val_Mid.append(MidPoint[j])
                    Error_Quantize.append(round((MidPoint[j]-y),3))
                    d=j
                    Encoded.append(format(d, f'0{BitsNum}b'))
                    break

        for i in y:
            Index_and_Mid(i)
        #Step6
        result=0
        for m in Error_Quantize:
            result=result+np.power(m,2)
        Avg_PowerE=result/len(Error_Quantize)

        if typ == "Bits":
            Test1.QuantizationTest1("quan1_out.txt",Encoded,Val_Mid)
        else:
            Test2.QuantizationTest2("quan2_out.txt",Val_Index,Encoded,Val_Mid,Error_Quantize)

        print(Val_Mid)
        print(Error_Quantize)
        print(Encoded)
        print(Val_Index)
        print(InterVals)
        print(MidPoint)


    Quantize_button=Button(Quantize_window,text="Quantize",bg="pink",fg="blue",command=Quantize)
    Quantize_button.grid(row=2,column=1)

#Task4
def File_creation(amplitud,phases,pth):

    with open(pth,'w') as File:
        File.write(format(0))
        File.write("\n")
        File.write(format(1))
        File.write("\n")
        File.write(format(len(amplitud)))
        for ampl,phse in zip(amplitud,phases):
            ampl=str(ampl)
            phse=str(phse)
            File.write("\n"+ampl+" "+phse)
def DFT_and_IDFT_Modify():
    DFTandIDFT_window=Toplevel(root)
    DFTandIDFT_window.title("Discrete Fourier Transform and Inverse")
    DFTandIDFT_leble=Label(DFTandIDFT_window,text="Choose Type of operation on the signal DFT or IDFT Modify_Component")
    DFTandIDFT_leble.grid(row=0,column=0)
    DFTandIDFT_ComboBox=ttk.Combobox(DFTandIDFT_window,values=("DFT","IDFT","Modify_Component"),state="raeadonly")
    DFTandIDFT_ComboBox.grid(row=0,column=1)
    DFTandIDFT_leble2=Label(DFTandIDFT_window,text="Enter Frequancy (fs)")
    DFTandIDFT_leble2.grid(row=1,column=0)
    DFTandIDFT_entry=Entry(DFTandIDFT_window)
    DFTandIDFT_entry.grid(row=1,column=1)
    amplitud=[]
    phases=[]
    def DFT_IDFT_Modify():
        typ=DFTandIDFT_ComboBox.get()
        if typ=="DFT":
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(Fpath)
            DFT(y)
        elif typ=="IDFT":
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(Fpath)
            IDFT(x,y)
        elif typ=="Modify_Component":
            Modify_Component()
    def DFT(y):
        fs=float(DFTandIDFT_entry.get())
        Lngth=len(y)
        Limagin=[]
        Lreal=[]
        X_Axis_multiples=[]
        for i in range(Lngth):
            imagin=0
            real=0
            for j in range(Lngth):
                Angl=(2*(np.pi*i*j))/Lngth
                imagin=imagin-y[j]*np.sin(Angl)
                real=real+y[j]*np.cos(Angl)
            Limagin.append(imagin)
            Lreal.append(real)
            ampl=np.sqrt(np.power(real,2) + np.power(imagin,2))
            amplitud.append(ampl)
            rad_ph=np.arctan2(imagin,real)
            degree_ph=np.degrees(rad_ph)
            phases.append(rad_ph)
            Axis_X=(2*np.pi*fs)/Lngth
            x_multiples=(Axis_X*(i+1))
            X_Axis_multiples.append(x_multiples)

        plt.figure()
        print(X_Axis_multiples)
        plt.plot(X_Axis_multiples,amplitud)
        plt.title("DFT Signal")
        plt.xlabel('Frequancy')
        plt.ylabel('Amplitude')
        plt.show()
        ##############
        plt.figure()
        plt.plot(X_Axis_multiples,phases)
        plt.title("DFT Signal")
        plt.xlabel('Frequancy')
        plt.ylabel('phase')
        plt.show()
        File_creation(amplitud,phases,"Polar_Form.txt")
        path="Output_Signal_DFT_A,Phase.txt"
        a,p=read_signal_from_file(path)
        am=signalcompare.SignalComapreAmplitude(amplitud,a)
        ph=signalcompare.SignalComaprePhaseShift(phases,p)
        print(am)
        print(ph)
        print(amplitud)
        print(phases)
        print(Lreal)
        print(Limagin)
    def IDFT(x,y):
            RealL=[]
            ImaginL=[]
            RealL2=[]
            for i in range(len(x)):
                RealL.append(x[i]*np.cos(y[i]))
                ImaginL.append(x[i]*np.sin(y[i]))
            Lngth=len(RealL)
            for i in range(Lngth):
                SUM=0
                for j in range(Lngth):
                    Angl=(2*(np.pi*i*j))/Lngth
                    imagin=np.sin(Angl)
                    real=np.cos(Angl)
                    SUM=SUM+((real*RealL[j])-(imagin*ImaginL[j]))

                RealL2.append(round(SUM/Lngth))
            path="Output_Signal_IDFT.txt"
            n,a=read_signal_from_file(path)
            am=signalcompare.SignalComapreAmplitude(RealL2,a)
            print(am)
            print(RealL2)
    def Modify_Component():
        Modify_window=Toplevel(DFTandIDFT_window)
        Modify_window.title("Modify")
        Modify_leble=Label(Modify_window,text="Enter The Index you want to Modify")
        Modify_leble.grid(row=0,column=0)
        Index=Entry(Modify_window)
        Index.grid(row=0,column=1)
        Modify_leble2=Label(Modify_window,text="Enter The Amplitude")
        Modify_leble2.grid(row=1,column=0)
        Amplitude=Entry(Modify_window)
        Amplitude.grid(row=1,column=1)
        Modify_leble3=Label(Modify_window,text="Enter The Phases")
        Modify_leble3.grid(row=2,column=0)
        Phase=Entry(Modify_window)
        Phase.grid(row=2,column=1)
        def Modify():
            indx=int(Index.get())
            Amplitud=Amplitude.get()
            Phas=Phase.get()
            with open("Polar_Form.txt","r") as Fpth:
                lins=Fpth.readlines()
            if 0<=indx<len(lins):
                lins[indx+3]=f"{Amplitud} {Phas}\n"
                with open("Polar_Form.txt","w") as Fpth:
                    Fpth.writelines(lins)
            else:
                print("Invalid Index")
        Modify_button=Button(Modify_window,text="Modify",bg="pink",fg="blue",command=Modify)
        Modify_button.grid(row=4,column=1)

    DFTandIDFT_button=Button(DFTandIDFT_window,text="Enter",bg="pink",fg="blue",command=DFT_IDFT_Modify)
    DFTandIDFT_button.grid(row=3,column=1)

#Task5
def DCTandRemoveDC():
    DCTandRemoveDC_window=Toplevel(root)
    DCTandRemoveDC_window.title("DCTandRemoveDC")
    DCTandRemoveDC_leble=Label(DCTandRemoveDC_window,text="Choose DCT or Remove DC")
    DCTandRemoveDC_leble.grid(row=0,column=0)
    DCTandRemoveDC_ComboBox=ttk.Combobox(DCTandRemoveDC_window,values=("DCT","Remove_DC"),state="raeadonly")
    DCTandRemoveDC_ComboBox.grid(row=0,column=1)

    def choose():
        typ=DCTandRemoveDC_ComboBox.get()
        if typ=="DCT":
            DCT()
        elif typ=="Remove_DC":
            RemoveDC()

    def DCT():
        DCT_window=Toplevel(DCTandRemoveDC_window)
        DCT_window.title("DCT")
        DCT_leble2=Label(DCT_window,text="Enter Number of Coefficients")
        DCT_leble2.grid(row=1,column=0)
        DCT_entry=Entry(DCT_window)
        DCT_entry.grid(row=1,column=1)
        def dct():
            Coefficients=int(DCT_entry.get())
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(Fpath)
            lenth=len(y)
            dct_vals=[]
            for i in range(lenth):
                sum=0
                for j in range(lenth):
                    ANG=(np.pi/(4*lenth))*((2*i)-1)*((2*j)-1)
                    sum=sum+(np.cos(ANG)*y[j])
                result=np.sqrt(2/lenth)*sum
                dct_vals.append(round(result,4))
            Creat_DCT_File(Coefficients,lenth,dct_vals)
            comparesignal2.SignalSamplesAreEqual("DCT_output.txt",dct_vals)
            print(dct_vals)
        dct_button=Button(DCT_window,text="compute",bg="pink",fg="blue",command=dct)
        dct_button.grid(row=2,column=1)

    def RemoveDC():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(Fpath)
        sum = 0
        for i in y:
            sum = sum +i
        avg=sum/len(x)
        for j in range(len(x)):
           y[j]=round(y[j]-avg,3)
        comparesignal2.SignalSamplesAreEqual("DC_component_output.txt",y)
        print(y)

    def Creat_DCT_File(Coefficients,n,VDCT):
        pth="DCT_File.txt"
        with open(pth,'w') as File:
            File.write(format(0))
            File.write("\n")
            File.write(format(1))
            File.write("\n")
            if 0<Coefficients<n:
                File.write(format(Coefficients))
                for i in range(Coefficients):
                    File.write("\n"+str(VDCT[i]))
            else:
                print("Please Enter a Valid Coefficient")

    DCTandRemoveDC_button=Button(DCTandRemoveDC_window,text="Enter",bg="pink",fg="blue",command=choose)
    DCTandRemoveDC_button.grid(row=1,column=1)

#Task6
def Named_Time_Domain():
    NamedTimeDomain_window=Toplevel(root)
    NamedTimeDomain_window.title("Named Time Domain for Signals")
    NamedTimeDomain_leble=Label(NamedTimeDomain_window,text="Choose Type of operation of Named Time Domain")
    NamedTimeDomain_leble.grid(row=0,column=0)
    NamedTimeDomain_ComboBox=ttk.Combobox(NamedTimeDomain_window,values=("Smoothing","Sharpening","Delaying||advancing",
                "Floding","DA for Floded signal","RemoveDC Frequancy domain","Convelution"),state="raeadonly")
    NamedTimeDomain_ComboBox.grid(row=0,column=1)

    def choose():
        typ=NamedTimeDomain_ComboBox.get()
        if typ=="Smoothing":
            Smoothing_Signal()
        elif typ=="Sharpening":
            DerivativeSignal.DerivativeSignal()
        elif typ=="Delaying||advancing":
            Delayingoradvancing()
        elif typ=="Floding":
            Floding_Signals()
        elif typ=="DA for Floded signal":
            DA_Floding()
        elif typ=="RemoveDC Frequancy domain":
            RemoveDC_Fdomain()
        elif typ=="Convelution":
            Convelution()

    def Smoothing_Signal():
        Smoothing_window=Toplevel(NamedTimeDomain_window)
        Smoothing_window.title("Smoothing")
        Smoothing_leble=Label(Smoothing_window,text="Enter number of points in averaging")
        Smoothing_leble.grid(row=0,column=0)
        Smoothing_Entry=Entry(Smoothing_window)
        Smoothing_Entry.grid(row=0,column=1)
        def Smoothing():
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(Fpath)
            Size=int(Smoothing_Entry.get())
            lenth=len(x)
            smoothed=[]
            for s in range(lenth):
                e=(Size-1)+s
                if lenth<=e:
                    break
                REP_Size=y[s:e+1]
                Avreage=np.sum(REP_Size)/Size
                smoothed.append(Avreage)
            if Size==3:
                result=comparesignal2.SignalSamplesAreEqual("OutMovAvgTest1.txt",smoothed)
                print(result)
            elif Size==5:
                result2=comparesignal2.SignalSamplesAreEqual("OutMovAvgTest2.txt",smoothed)
                print(result2)
            print(smoothed)
            return smoothed

        Smoothing_button=Button(Smoothing_window,text="Smoothe",bg="pink",fg="blue",command=Smoothing)
        Smoothing_button.grid(row=2,column=1)

    def Delaying_advancing(path,number):
        x, y = read_signal_from_file(path)
        numbr=int(number)
        listDelaying=[i+numbr for i in x]
        print(listDelaying)
        return listDelaying
    def Delayingoradvancing():
        Delayingoradvancing_window=Toplevel(NamedTimeDomain_window)
        Delayingoradvancing_window.title("Delaying & advancing")
        Delayingoradvancing_leble=Label(Delayingoradvancing_window,text="Choose Delaying or Advancing for Signal")
        Delayingoradvancing_leble.grid(row=0,column=0)
        Delayingoradvancing_Entry=Entry(Delayingoradvancing_window)
        Delayingoradvancing_Entry.grid(row=0,column=1)
        def DA():
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            Delaying_advancing(Fpath,Delayingoradvancing_Entry.get())
        s_button=Button(Delayingoradvancing_window,text="Enter",bg="pink",fg="blue",command=DA)
        s_button.grid(row=3,column=1)

    def Floding_Signals():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(Fpath)
        y.reverse()
        File_creation(x,y,"flod.txt")
        print(y)
        return y

    def DA_Floding():
        DA_Floding_window=Toplevel(NamedTimeDomain_window)
        DA_Floding_window.title("Delaying & advancing for Floded Signals")
        DA_Floding_leble=Label(DA_Floding_window,text="enter shifting number")
        DA_Floding_leble.grid(row=0,column=0)
        DA_Floding_Entry=Entry(DA_Floding_window)
        DA_Floding_Entry.grid(row=0,column=1)
        def run():
            num=DA_Floding_Entry.get()
            y=Floding_Signals()
            arr= Delaying_advancing("flod.txt",num)
            if int(num)>0:
                Shift_Fold_Signal.Shift_Fold_Signal("Output_ShifFoldedby500.txt",arr,y)
            elif int(num)<0:
                Shift_Fold_Signal.Shift_Fold_Signal("Output_ShiftFoldedby-500.txt",arr,y)
        Smoothing_button=Button(DA_Floding_window,text="enter",bg="pink",fg="blue",command=run)
        Smoothing_button.grid(row=2,column=1)

    def RemoveDC_Fdomain():
        amplitudes, phases= new_DFT()
        print(amplitudes)
        print(phases)
        amplitudes[0]=0
        phases[0]=0
        Lout=new_IDFT(amplitudes,phases)
        for i in range(len(Lout)):
            Lout[i]=round(Lout[i],3)
        comparesignal2.SignalSamplesAreEqual("DC_component_output.txt",Lout)
        print(Lout)
        return Lout
    def new_DFT():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(Fpath)
        Lngth=len(y)
        Limagin=[]
        Lreal=[]
        amplitud=[]
        phases=[]
        for i in range(Lngth):
            imagin=0
            real=0
            for j in range(Lngth):
                Angl=(2*(np.pi*i*j))/Lngth
                imagin=imagin-y[j]*np.sin(Angl)
                real=real+y[j]*np.cos(Angl)
            Limagin.append(imagin)
            Lreal.append(real)
            ampl=np.sqrt(np.power(real,2) + np.power(imagin,2))
            amplitud.append(ampl)
            rad_ph=np.arctan2(imagin,real)
            phases.append(rad_ph)
        return amplitud,phases
    def new_IDFT(x,y):
        RealL=[]
        ImaginL=[]
        RealL2=[]
        for i in range(len(x)):
            RealL.append(x[i]*np.cos(y[i]))
            ImaginL.append(x[i]*np.sin(y[i]))
        l=len(x)
        for i in range(l):
            Sum=0
            for j in range(l):
                Angl=(2*(np.pi*i*j))/l
                imagin=np.sin(Angl)
                real=np.cos(Angl)
                Sum=Sum+((real*RealL[j])-(imagin*ImaginL[j]))
            RealL2.append(Sum/l)
        print(RealL2)
        return RealL2

    def Convelution():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x,y= read_signal_from_file(Fpath)
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x2,y2= read_signal_from_file(path)
        l1=len(x);l2=len(x2);cov=[]
        for i in range(l1):
            for j in range(l2):
                cov.append(0)
        for i in range(l1):
            for j in range(l2):
                cov[i+j]+=int(y[i]*y2[j])
        con=[]
        start=int(x[0]+x2[0])
        end=int(x[-1]+x2[-1])
        for h in range(start,end+1):
            con.append(h)
        print(con)
        print(cov)
        ConvTest.ConvTest(con,cov)

    choose_button=Button(NamedTimeDomain_window,text="Enter",bg="pink",fg="blue",command=choose)
    choose_button.grid(row=3,column=1)

#Task7
def read_signal_from_file_corr(file_path):
        with open(file_path) as file:
            listOfLines=file.readlines()
            samplesX = []
            for line in listOfLines:
                if "," in line:
                    if "f" in line:
                        splitList = line.strip().split(",")
                        splitList[0]=splitList[0].replace("f","")
                    else:
                        splitList = line.strip().split(",")
                else:
                    if "f" in line:
                        splitList = line.strip().split()
                        splitList[0]=splitList[0].replace("f","")
                    else:
                        splitList = line.strip().split()
                sampleX = float(splitList[0])
                samplesX.append(sampleX)
        return samplesX
def Correlation_TimeAnalysis_Tempmatching():
    CorrTAnalysisTempM_window=Toplevel(root)
    CorrTAnalysisTempM_window.title("Correlation , TimeAnalysis , Template Matching for Signals")
    CorrTAnalysisTempM_leble=Label(CorrTAnalysisTempM_window,
                                   text="Choose Type of operation of (Correlation , TimeAnalysis , Template Matching)")
    CorrTAnalysisTempM_leble.grid(row=0,column=0)
    CorrTAnalysisTempM_ComboBox=ttk.Combobox(CorrTAnalysisTempM_window,values=("Correlation","Time Analysis",
                                                                    "Template matching"),state="raeadonly")
    CorrTAnalysisTempM_ComboBox.grid(row=0,column=1)
    def choose():
        typ=CorrTAnalysisTempM_ComboBox.get()
        if typ=="Correlation":
            ind,cor=Correlation()
            CompareSignal.Compare_Signals("CorrOutput.txt",ind,cor)
        elif typ=="Time Analysis":
            Time_Analysis()
        elif typ=="Template matching":
            Template_matching()
    def Correlation():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        index,X= read_signal_from_file(Fpath)
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        index2,X2= read_signal_from_file(path)
        S1=0 ; S2=0 ; r=[];P=[];indxs=[]; lengh=len(index)
        for i in range(lengh):
            S1+=np.power(X[i],2)
        for j in range(len(index2)):
            S2+=np.power(X2[j],2)
        Maqam=(np.power((S1*S2),0.5))/lengh
        for l in range(lengh):
            Sum =0
            for k in range(lengh):
                Sum+=(X[k]*X2[k])
            r.append(Sum/lengh)
            Fval=X2[0]
            X2.remove(X2[0])
            X2.append(Fval)
        for l in range(lengh):
            indxs.append(l)
            P.append(round(r[l]/Maqam,8))
        print(P)
        return indxs,P
    def Time_Analysis():
        TimeAnalysis_window=Toplevel(CorrTAnalysisTempM_window)
        TimeAnalysis_window.title("TimeAnalysis for Signals")
        TimeAnalysis_leble=Label(TimeAnalysis_window,text="Enter Fs )")
        TimeAnalysis_leble.grid(row=0,column=0)
        TimeAnalysis_entry=Entry(TimeAnalysis_window)
        TimeAnalysis_entry.grid(row=0,column=1)
        def TimeAnalysis():
            Fs=float(TimeAnalysis_entry.get())
            indxs,P=Correlation()
            corr=max(P)
            for i in range(len(P)):
                if P[i]==corr:
                    j=i
                    break
            print("Time Delay : "+str(j)+'/'+str(Fs))
            print(corr)

        Enter_button=Button(TimeAnalysis_window,text="Enter",bg="pink",fg="blue",command=TimeAnalysis)
        Enter_button.grid(row=3,column=1)

    def Template_matching():
        Avg1=[];Avg2=[];Files1=[];Files2=[];r=[];r2=[]
        f=glob.glob("Class 2/*.txt")
        for n in range(len(f)):
            file1="Class 1/down"+str(n+1)+".txt"
            vals1=read_signal_from_file_corr(file1)
            Files1.append(vals1)
            file2="Class 2/up"+str(n+1)+".txt"
            vals2=read_signal_from_file_corr(file2)
            Files2.append(vals2)
        for i in range(len(Files1[0])):
            sm1=0;sm2=0
            for j in range(len(Files1)):
                sm1+=Files1[j][i]
                sm2+=Files2[j][i]
            Avg1.append(sm1/len(Files1[0]))
            Avg2.append(sm2/len(Files2[0]))
        Test = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        vals=read_signal_from_file_corr(Test)
        lngh=len(vals)
        for l in range(lngh):
            Sum =0;Sum2 =0
            for k in range(lngh):
                Sum+=(Avg1[k]*vals[k])
                Sum2+=(Avg2[k]*vals[k])
            r.append(Sum/lngh)
            r2.append(Sum2/lngh)
            Fval=vals[0]
            vals.remove(vals[0])
            vals.append(Fval)
        mr=max(r);mr2=max(r2)
        if mr > mr2:
            print("Predicted Calss of Signal is down")
        elif mr2 > mr:
            print("Predicted Calss of Signal is up")
    Enter_button=Button(CorrTAnalysisTempM_window,text="Enter",bg="pink",fg="blue",command=choose)
    Enter_button.grid(row=3,column=1)

#Task8
def Fast_Convelution_and_Correlation():
    Fast_Conv_and_Corr_window=Toplevel(root)
    Fast_Conv_and_Corr_window.title("Named Time Domain for Signals")
    Fast_Conv_and_Corr_leble=Label(Fast_Conv_and_Corr_window,
                                   text="Choose Type of operation Fast Convelution orFast Correlation")
    Fast_Conv_and_Corr_leble.grid(row=0,column=0)
    Fast_Conv_and_Corr_ComboBox=ttk.Combobox(Fast_Conv_and_Corr_window,
                                             values=("Fast Convelution","Fast Correlation"),state="raeadonly")
    Fast_Conv_and_Corr_ComboBox.grid(row=0,column=1)

    def choose():
        typ=Fast_Conv_and_Corr_ComboBox.get()
        if typ=="Fast Convelution":
            Fast_Convelution()
        elif typ=="Fast Correlation":
            Fast_Correlation()
    def dft(rel):
        length=len(rel)
        comp= np.zeros(length, dtype=complex)
        for i in range(length):
            comp[i]=0
            for l in range(length):
                Exp=np.exp((-2j * np.pi * i * l )/ length)
                comp[i]=comp[i]+(rel[l] * Exp)
        return comp
    def idft(complx):
        length=len(complx)
        real=[]
        for i in range(length):
            rl=0
            for l in range(length):
                Ang=(2 * np.pi * i * l )/ length
                I=np.sin(Ang)
                R=np.cos(Ang)
                rl=rl+((R*complx[l].real)-(I*complx[l].imag))
            real.append(rl/length)
        return real
    def Fast_Convelution():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x,y= read_signal_from_file(Fpath)
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x2,y2= read_signal_from_file(path)
        l1=len(x);l2=len(x2);FREQdomain=[]
        EXTDlen = l1 + l2 - 1
        Ypad1 = np.pad(y, (0, EXTDlen - l1))
        Ypad2 = np.pad(y2, (0, EXTDlen - l2))
        Yc1=dft(Ypad1)
        Yc2=dft(Ypad2)
        for i in range(len(Yc1)):
            FREQdomain.append(Yc1[i]*Yc2[i])
        TIMdomain = idft(FREQdomain)
        print(TIMdomain)
        indecies = np.arange(int(min(x)), int(min(x)) + EXTDlen)
        ConvTest.ConvTest(indecies, TIMdomain)
        return indecies, TIMdomain
    def Fast_Correlation():
        Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x,y= read_signal_from_file(Fpath)
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x2,y2= read_signal_from_file(path)
        l=len(y);FREQdomain=[];Rres=[]
        Yc1=dft(y)
        Yc2=dft(y2)
        ycong=Yc1.conjugate()
        for i in range(len(Yc1)):
            FREQdomain.append(ycong[i]*Yc2[i])
        TIMdomain=idft(FREQdomain)
        for i in range(l):
            Rres.append(TIMdomain[i]/l)
        indeceis=np.arange(0,l)
        CompareSignal.Compare_Signals("Corr_Output.txt", indeceis,Rres)

    Enter_button=Button(Fast_Conv_and_Corr_window,text="Enter",bg="pink",fg="blue",command=choose)
    Enter_button.grid(row=3,column=1)

#Task_Practical
def Practical():
    Practical_window=Toplevel(root)
    Practical_window.title("Practical")
    Practical_leble=Label(Practical_window,text="Choose Practica_Task1 or Practica_Task2")
    Practical_leble.grid(row=0,column=0)
    Practical_ComboBox=ttk.Combobox(Practical_window,values=("Practica_Task1","Practica_Task2"),state="raeadonly")
    Practical_ComboBox.grid(row=0,column=1)
    def choose():
        typ=Practical_ComboBox.get()
        if typ=="Practica_Task1":
            Practical_1()
        elif typ=="Practica_Task2":
            Practical_2()

    def Practical_1():
        Practical_1_window=Toplevel(Practical_window)
        Practical_1_window.title("Practica Task 1")
        Practical_T1_leble=Label(Practical_1_window,text="Choose Filtering or Resampling")
        Practical_T1_leble.grid(row=0,column=0)
        Practical_T1_ComboBox=ttk.Combobox(Practical_1_window,values=("Filtering","Resampling"),state="raeadonly")
        Practical_T1_ComboBox.grid(row=0,column=1)
        def chos():
            typ=Practical_T1_ComboBox.get()
            print(typ)
            if typ=="Filtering":
                Filtering()
            elif typ=="Resampling":
                Resampling()
        def Filtering():
            Filtering_window=Toplevel(Practical_1_window)
            Filtering_window.title("Filtering")
            Filtering_leble=Label(Filtering_window,text="Choose Filter Type")
            Filtering_leble.grid(row=0,column=0)
            Filtering_ComboBox=ttk.Combobox(Filtering_window,
                                values=("Low_pass","High_pass","Band_pass","Band_stop"),state="raeadonly")
            Filtering_ComboBox.grid(row=0,column=1)
            FS_leble=Label(Filtering_window,text="Enter FS");FS_leble.grid(row=1,column=0)
            FS_Entry=Entry(Filtering_window);FS_Entry.grid(row=1,column=1)
            StopBandAttenuation_leble=Label(Filtering_window,text="Enter Stop band attenuation");StopBandAttenuation_leble.grid(row=2,column=0)
            StopBandAttenuation_Entry=Entry(Filtering_window);StopBandAttenuation_Entry.grid(row=2,column=1)
            FC_leble=Label(Filtering_window,text="Enter FC");FC_leble.grid(row=3,column=0)
            FC_Entry=Entry(Filtering_window);FC_Entry.grid(row=3,column=1)
            TransitionBand_leble=Label(Filtering_window,text="Enter Transition band");TransitionBand_leble.grid(row=4,column=0)
            TransitionBand_Entry=Entry(Filtering_window);TransitionBand_Entry.grid(row=4,column=1)
            FC2_leble=Label(Filtering_window,text="Enter FS2 if you chose band pass or stop");FC2_leble.grid(row=5,column=0)
            FC2_Entry=Entry(Filtering_window)
            FC2_Entry.insert(0, "0")
            FC2_Entry.grid(row=5,column=1)

            x = []
            y = []
            def get_input_file ():
                nonlocal x, y
                Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
                x,y= read_signal_from_file(Fpath)
                
            Import_leble=Label(Filtering_window,text="Input File");Import_leble.grid(row=6,column=0)
            Import_Button=Button(Filtering_window, text="Import", command=get_input_file)
            Import_Button.grid(row=6,column=1)
            def filter():
                FT=Filtering_ComboBox.get();Fs=float(FS_Entry.get())
                Stop_Band_Attenuation=float(StopBandAttenuation_Entry.get())
                Fc=float(FC_Entry.get());Transition_Band=float(TransitionBand_Entry.get())
                Fc2=float(FC2_Entry.get())
                Indices, Result_Filter = Design_FIR_FILTER(FT,Fs,Stop_Band_Attenuation,Fc,Transition_Band,Fc2)
                plt.figure()
                plt.plot(Indices,Result_Filter)
                plt.title("Filter Coefficients")
                plt.xlabel('Indices')
                plt.ylabel('Ampl')
                plt.show()
                if len(x) > 0:
                    Indices,Result_Filter=Apply_Filter(x,y,Indices,Result_Filter)
                    plt.figure()
                    plt.plot(x,y)
                    plt.title("Input Signal")
                    plt.xlabel('Indices')
                    plt.ylabel('Ampl')
                    plt.show()
                    plt.figure()
                    plt.plot(Indices,Result_Filter)
                    plt.title("Filtered Signal")
                    plt.xlabel('Indices')
                    plt.ylabel('Ampl')
                    plt.show()
                File_creation(Indices,Result_Filter,"FilterCoefficients.txt")
                Path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
                CompareSignal.Compare_Signals(Path,Indices,Result_Filter)
            Enter_button=Button(Filtering_window,text="Enter",bg="pink",fg="blue",command=filter)
            Enter_button.grid(row=7,column=1)
        def Resampling():
            Resampling_window=Toplevel(Practical_1_window)
            Resampling_window.title("Resampling")
            FS_leble=Label(Resampling_window,text="Enter FS");FS_leble.grid(row=1,column=0)
            FS_Entry=Entry(Resampling_window);FS_Entry.grid(row=1,column=1)
            StopBandAttenuation_leble=Label(Resampling_window,text="Enter Stop Band Attenuation")
            StopBandAttenuation_leble.grid(row=2,column=0)
            StopBandAttenuation_Entry=Entry(Resampling_window);StopBandAttenuation_Entry.grid(row=2,column=1)
            FC_leble=Label(Resampling_window,text="Enter FC");FC_leble.grid(row=3,column=0)
            FC_Entry=Entry(Resampling_window);FC_Entry.grid(row=3,column=1)
            TransitionBand_leble=Label(Resampling_window,text="Enter Transition Band");TransitionBand_leble.grid(row=4,column=0)
            TransitionBand_Entry=Entry(Resampling_window);TransitionBand_Entry.grid(row=4,column=1)
            UF_leble=Label(Resampling_window,text="Enter Upsampling Factor");UF_leble.grid(row=5,column=0)
            UF_Entry=Entry(Resampling_window);UF_Entry.grid(row=5,column=1)
            DF_leble=Label(Resampling_window,text="Enter Downsampling Factor");DF_leble.grid(row=6,column=0)
            DF_Entry=Entry(Resampling_window);DF_Entry.grid(row=6,column=1)
            def resample():
                Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
                x,y= read_signal_from_file(Fpath)
                Fs=float(FS_Entry.get());Stop_Band_Attenuation=float(StopBandAttenuation_Entry.get())
                Fc=float(FC_Entry.get());Transition_Band=float(TransitionBand_Entry.get())
                UF=int(UF_Entry.get());DF=int(DF_Entry.get())
                resamplex,resampley=Resample_Signal(x,y,"Low_pass",Fs,Stop_Band_Attenuation,Fc,Transition_Band,UF,DF)
                plt.figure()
                plt.plot(resamplex,resampley)
                plt.title("Resampling Result")
                plt.xlabel('Indices')
                plt.ylabel('Ampl')
                plt.show()
                Path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
                File_creation(resamplex,resampley,"Resample.txt")
                CompareSignal.Compare_Signals(Path,resamplex,resampley)
                print(resamplex,resampley)

            Enter_button=Button(Resampling_window,text="Enter",bg="pink",fg="blue",command=resample)
            Enter_button.grid(row=8,column=1)

        Enter_button=Button(Practical_1_window,text="Enter",bg="pink",fg="blue",command=chos)
        Enter_button.grid(row=3,column=1)

    def Practical_2():
        Practical_2_window=Toplevel(Practical_window)
        Practical_2_window.title("Practica Task 2")
        Practical_2_leble=Label(Practical_2_window,text="Enter FS");Practical_2_leble.grid(row=0,column=0)
        FS_Entry=Entry(Practical_2_window);FS_Entry.grid(row=0,column=1)
        Fmax_leble=Label(Practical_2_window,text="Enter F max");Fmax_leble.grid(row=1,column=0)
        Fmax_Entry=Entry(Practical_2_window);Fmax_Entry.grid(row=1,column=1)
        Fmin_leble=Label(Practical_2_window,text="Enter F min");Fmin_leble.grid(row=2,column=0)
        Fmin_Entry=Entry(Practical_2_window);Fmin_Entry.grid(row=2,column=1)
        Fnew_leble=Label(Practical_2_window,text="Enter FS new");Fnew_leble.grid(row=3,column=0)
        Fnew_Entry=Entry(Practical_2_window);Fnew_Entry.grid(row=3,column=1)
        def run():
            NewFs=float(Fnew_Entry.get());Fs=int(FS_Entry.get())
            Files1=[];Files2=[]
            f=glob.glob("A/*.txt")
            for n in range(len(f)):
                file1="A/ASeg"+str(n+1)+".txt"
                vals1=read_signal_from_file_corr(file1)
                Files1.append(vals1)
                file2="B/BSeg"+str(n+1)+".txt"
                vals2=read_signal_from_file_corr(file2)
                Files2.append(vals2)
            Fpath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            y= read_signal_from_file_corr(Fpath)
            plt.figure()
            plt.plot(range(len(y)),y)
            plt.title("Original Signal")
            plt.xlabel('Indices')
            plt.ylabel('Ampl')
            plt.show()
            Fx,Fy=Design_FIR_FILTER("Band_pass",int(Fmin_Entry.get()),50,0,500,int(Fmax_Entry.get()))

            Rex,Rey=Apply_Filter(range(len(y)),y,Fx,Fy)

            if NewFs >= 2 * Fs:
                #Ds=int(NewFs/Fs);Us=int(Fs/NewFs)
                Rex,Rey=Resample_Signal(Rex,Rey,"Low_pass",Fs,53,0,500,3,2)
            else:
                messagebox.showerror(" newFs is not valid")

            Rey=Remove_dc(Rey)

            Rey=Normalize(Rey)

            Rey=Cros_correlation(Rey,Rey)
            plt.figure()
            plt.plot(Rex,Rey)
            plt.title("After Auto-correlation")
            plt.xlabel('Indices')
            plt.ylabel('Ampl')
            plt.show()

            Rey=DCT(Rey)
            plt.figure()
            plt.plot(Rex,Rey)
            plt.title("After DCT")
            plt.xlabel('Indices')
            plt.ylabel('Ampl')
            plt.show()

            matching_res=classify_Template(Rey,Files1,Files2)
            print("Template Matching Result:\n", matching_res)
            # print(Files1)
            # print(Files2)

        Enter_button=Button(Practical_2_window,text="Enter",bg="pink",fg="blue",command=run)
        Enter_button.grid(row=4,column=1)

    def Design_FIR_FILTER(Filter_Type, Fs, Stop_Band_Attenuation, Fc, Transition_Band, Fc2=None):
        Delta=Transition_Band/Fs;n=0
        HoldFilters=[];Indices=[]
        #Blackman
        if Stop_Band_Attenuation <= 21:
            RonddNum = math.ceil(0.9/Delta)
            if RonddNum % 2 == 0:
                RonddNum += 1
            n = RonddNum
        elif Stop_Band_Attenuation <= 44:
            RonddNum = math.ceil(3.1/Delta)
            if RonddNum % 2 == 0:
                RonddNum += 1
            n = RonddNum
        elif Stop_Band_Attenuation <= 53:
            RonddNum = math.ceil(3.3/Delta)
            if RonddNum % 2 == 0:
                RonddNum += 1
            n = RonddNum
        # Hanning
        elif Stop_Band_Attenuation <= 74:
            RonddNum = math.ceil(5.5/Delta)
            if RonddNum % 2 == 0:
                RonddNum += 1
            n = RonddNum

        Indices=range(-math.floor(n/2), math.floor(n/2) + 1)
        #Bring Filter
        if Filter_Type=="Low_pass":
            Nfc=(Fc+(Transition_Band/2))/Fs
            for i in Indices:
                Window=Window_Size(Stop_Band_Attenuation,i,n)
                if i==0:
                    hld=Nfc*2
                else:
                    cof=np.pi*Nfc*i*2
                    hld=2*Nfc*(np.sin(cof)/cof)
                HoldFilters.append(hld*Window)
        elif Filter_Type=="High_pass":
            Nfc=(Fc-(Transition_Band/2))/Fs
            for i in Indices:
               Window = Window_Size(Stop_Band_Attenuation,i,n)
               if i==0:
                   hld=1-(2*Nfc)
               else:
                   cof=i*2*Nfc*np.pi
                   hld=-2*Nfc*(np.sin(cof)/(cof))
               HoldFilters.append(hld*Window)
        elif Filter_Type=="Band_pass":
            Nfc=(Fc-(Transition_Band/2))/Fs
            Nfc2=(Fc2+(Transition_Band/2))/Fs
            for i in Indices:
                Window = Window_Size(Stop_Band_Attenuation,i,n)
                if i==0:
                    hld=2*(Nfc2-Nfc)
                else:
                    cof1=i*2*np.pi*Nfc
                    cof2=i*2*np.pi*Nfc2
                    hld=(2*Nfc2*(np.sin(cof2)/cof2)) - (2*Nfc*(np.sin(cof1)/cof1))
                HoldFilters.append(hld*Window)
        elif Filter_Type=="Band_stop":
            Nfc=(Fc+(Transition_Band/2))/Fs
            Nfc2=(Fc2-(Transition_Band/2))/Fs
            for i in Indices:
                Window=Window_Size(Stop_Band_Attenuation,i,n)
                if i==0:
                    hld=1-2*(Nfc2-Nfc)
                else:
                    cof1=2*i*np.pi*Nfc
                    cof2=2*i*np.pi*Nfc2
                    hld=(2*Nfc*(np.sin(cof1)/cof1))-(2*Nfc2*(np.sin(cof2)/cof2))
                HoldFilters.append(hld*Window)
        return Indices,HoldFilters  #""""""
    def Window_Size(stop_band,I,index):
         if stop_band <= 21:
             return 1
         elif stop_band <= 44:
             return (0.5+(0.5*np.cos((2*I*np.pi)/index)))
         elif stop_band <= 53:
             return (0.54+(0.46*np.cos((2*I*np.pi)/index)))
         elif stop_band <= 74:
             return (0.42 + (0.5 * np.cos(2 * np.pi * I / (index - 1))) + 0.08 * np.cos(4 * np.pi * I / (index - 1)))
    def Apply_Filter(x,y,xf,yf):
        l1=len(y);l2=len(yf);res=[];Xvals=[]
        StartIndex=int(min(x)+min(xf))
        EndIndex=int(max(x)+max(xf))
        Xvals=range(StartIndex,EndIndex+1)
        for i in range(l1+l2-1):
            sum=0
            for j in range(min(i,l1-1)+1):
                if l2>i-j>=0:
                    sum+=(y[j]*yf[i-j])
            res.append(sum)
        return Xvals,res
    
    def Resample_Signal(x,y,Filter_Type, Fs, Stop_Band_Attenuation, Fc, Transition_Band, Us,Ds):
        ConIndeces=[]
        if Us==0 and Ds>0:
            Fx,Fy=Design_FIR_FILTER(Filter_Type,Fs,Stop_Band_Attenuation,Fc,Transition_Band,None)
            Outx,Outy= Apply_Filter(x,y,Fx,Fy)
            Outx = Outx[::Ds]
            Minx=min(Outx)
            length=len(Outx)
            ConIndeces=range(Minx,Minx+length)
            return ConIndeces,Outy[::Ds]
        elif Us>0 and Ds==0:
            Upsampled_signal=USAMPLE(y,Us)
            Upsampledx=USAMPLE(x,Us)
            Upsampledx = list(range(min(Upsampledx), min(Upsampledx) + len(Upsampledx)))
            Fx,Fy=Design_FIR_FILTER(Filter_Type,Fs,Stop_Band_Attenuation,Fc,Transition_Band,None)
            Outx,Outy= Apply_Filter(Upsampledx,Upsampled_signal,Fx,Fy)
            return Outx,Outy
        elif Us>0 and Ds>0:
            Upsampled_signaly=USAMPLE(y,Us)
            Upsampled_signalx=USAMPLE(x,Us)
            Upsampled_signalx=list(range(min(Upsampled_signalx), min(Upsampled_signalx) + len(Upsampled_signalx)))
            Fx,Fy=Design_FIR_FILTER(Filter_Type,Fs,Stop_Band_Attenuation,Fc,Transition_Band,None)
            Outx,Outy= Apply_Filter(Upsampled_signalx,Upsampled_signaly,Fx,Fy)
            Outx = Outx[::Ds]
            Minx=min(Outx)
            length=len(Outx)
            ConIndeces=range(Minx,Minx+length)
            return ConIndeces,Outy[::Ds]
        else:
            return messagebox.showerror("Downsampling is 0 and Upsampling is 0")
    def USAMPLE(signal, factor):
        result = []
        for element in signal:
            result.extend([element] + [0] * (factor-1))
        for i in range(factor-1):
            result.pop()
        return result
    def Remove_dc(x):
        x_array = np.array([])
        mean = np.mean(x)
        for i in x:
            x_array = np.append(x_array, i - mean)
        return x_array
    def Normalize(signal):
        max_value = max(np.abs(signal))
        signalnormalized = signal
        if max_value != 0:
            signalnormalized = signal / max_value
        return signalnormalized
    def Cros_correlation(y1, y2):
        # print(y1)
        # print(y2)
        N = len(y1)
        results = []

        for n in range(N):
            sum = 0
            for j in range(N):
                sum += y1[j]*y2[(j+n) % N]
            results.append(((1/N)*sum))

        return results
    def DCT(x):
        l = len(x);results = np.array([])
        for k in range(l):
            Values = []
            for i in range(l):
                ks=(2 * k - 1);NW=(2 * i - 1)
                Values.append(x[i] * np.cos((np.pi / (4 * l)) * NW *ks ))
            results = np.append(results, np.sum(Values))
        results =results * np.sqrt(2 / l)
        return results
    
    def get_samples(files_contents):
        max_samples = max(len(content) for content in files_contents)
        get_samples = np.zeros(max_samples)

        for content in files_contents:
            get_samples[:len(content)] += content

        get_samples /= len(files_contents)
        return get_samples
    def calculate_mean_correlation(test_file, class_content):
        num_samples = min(len(test_file), len(class_content))
        correlation = np.corrcoef(test_file[:num_samples], class_content[:num_samples])[0, 1]
        return correlation
    def classify_Template(test_file, class1_content, class2_content):
        class1_content = get_samples(class1_content)
        class2_content = get_samples(class2_content)
        correlation_class1 = calculate_mean_correlation(test_file, class1_content)
        correlation_class2 = calculate_mean_correlation(test_file, class2_content)

        if correlation_class1 > correlation_class2:
            result_text = "\nTemplate matches A"
        else:
            result_text = "\nTemplate matches B"

        return result_text


    Enter_button=Button(Practical_window,text="Enter",bg="pink",fg="blue",command=choose)
    Enter_button.grid(row=3,column=1)

########################################
# Signal generation window (gui)
root = Tk()
root.title("Signal Processing Framework")
# Task1
generate_signal_from_file_button = Button(root, text="Generate Signal File",
                                          bg="pink",fg="blue", command=generate_wave_from_file)
generate_signal_from_file_button.pack()

generate_signal_wave_button = Button(root, text="Generate Signal Wave",
                                     bg="pink",fg="blue", command=signal_generation_menu_gui)
generate_signal_wave_button.pack()

#Task2
Operations_On_Signal=Button(root,text="Operations On Signals",bg="pink",fg="blue",command=OperationsOnSignals)
Operations_On_Signal.pack()

#Task3
Quantize_Signal=Button(root,text="Quantize Signals",bg="pink",fg="blue",command=Quantization_Signals)
Quantize_Signal.pack()

#Task4
DFT_and_IDFT_Modify_Signal=Button(root,text="DFT_IDFT_Modify Signals",bg="pink",fg="blue",command=DFT_and_IDFT_Modify)
DFT_and_IDFT_Modify_Signal.pack()

#Task5
DCTandRemoveDC_Button=Button(root,text="DCT and Remove DC",bg="pink",fg="blue",command=DCTandRemoveDC)
DCTandRemoveDC_Button.pack()

#Task6
Named_Time_Domain_Button=Button(root,text="Named Time Domain",bg="pink",fg="blue",command=Named_Time_Domain)
Named_Time_Domain_Button.pack()

#Task7
Cor_Tanalysis_Tempmatch_Button = Button(root, text="Correlation TimeAnalysis Tempmatching",
                                        bg="pink",fg="blue", command=Correlation_TimeAnalysis_Tempmatching)
Cor_Tanalysis_Tempmatch_Button.pack()

#Task8
Fast_Conv_and_Corr_Button = Button(root, text="Fast Correlation & Fast Correlation",
                                   bg="pink",fg="blue", command=Fast_Convelution_and_Correlation)
Fast_Conv_and_Corr_Button.pack()

#Practical
Practical_Button = Button(root, text="Practical",bg="pink",fg="blue", command=Practical)
Practical_Button.pack()

# Run the main event loop
root.mainloop()
