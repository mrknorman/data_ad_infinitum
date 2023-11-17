= Application to Gravitational Waves

#set math.equation(numbering: "(1)")
We have demonstrated that simple artificial neural networks can be used to classify input data drawn from a restricted distribution into a number of classes, $N$, with a high ($>99.9%$) degree of accuracy. Because we didn't design the network with any consideration for the dataset (besides its dimensionality), we can infer that this method should be general enough to classify data drawn from other distributions that contain discrete differentiable classes. It is not clear, however, which other distributions can be classified and what magnitude of the network is required to achieve a similar degree of accuracy. It is easy to imagine distributions that are considerably simpler than the MNSIT dataset and, conversely, ones that are much more complex. There may be a mathematical approach to determine the link between the distribution and required model complexity. One possible metric that touches upon this relation is the Rademacher complexity, $accent(cal(R), hat)_M$, given by 

$ accent(cal(R), hat)_M (H) = EE_in  [1/M sum_(i=1)^M epsilon_i h [accent(x, arrow)_i]] , $
 
where $M$ is the number of data points in a dataset $X = [accent(x_1, arrow), ..., accent(x_i, arrow), ..., accent(x_M, arrow)]$ where each point is a vector, $accent(x, arrow)$, in our case the input vectors of our training dataset, $accent(epsilon, arrow) = [accent(epsilon_1, arrow), ..., accent(epsilon_i, arrow), ..., accent(epsilon_M, arrow)]$ uniformly distributed in ${-1, +1}^m$, and $H$ is a real-valued function class, in our case the set of functions that can be approximated by our chosen neural network architecture. The Rademacher complexity is a measure of how well functions in $H$ can fit random noise in the data. A higher Rademacher complexity indicates that the function class can fit noise better, which implies a higher capacity to overfit the data. So, one approach to optimising the model would be to attempt to minimise this value whilst maximising model performance. More details about this metric and its use in defining the relationship between data samples and model complexity can be found at @model_complexity. Despite the existence of this metric, however, it would appear that there has not been substantial research into the link between dataset complexity and required model size @model_complexity_2, though it is possible that such a paper has been missed.

One method that we can use to explore this question is to find out the answer empirically. As we move from the MNIST dataset to distributions within gravitational-wave data science, the natural starting point is to repeat the previous experiments with gravitational-wave data, both as a comparison and as a baseline as we move forward. The following subsection will explore the possible areas for detection applications, describe the example datasets and their construction, and explore the results of repeating the previous experiments on gravitational-wave data, comparing our results to similar attempts from the literature. By the end of this chapter, we will have accumulated a large number of possible network, training, and data configurations. We will perform a series of tests to begin to narrow the search space of possible hyperparameters.

== Gravitational-Wave Classifiers

The scope of gravitational-wave data problems to which we can apply artificial neural network models is large. However, we shall limit our investigation to perhaps the simplest type of problem --- classification, the same type of problem that we have previously demonstrated with our classification of the MNIST dataset into discrete classes. Though classification is arguably the most straightforward problem available to us, it remains one of the most crucial --- before any other type of transient signal analysis can be performed, transients must first be identified.

There are several problems in gravitational-wave data analysis which can be approached through the use of classification methods. These can broadly be separated into two classes of problems --- detection and differentiation. *Detection* problems are self-explanatory; these kinds of problems require the identification of the presence of features within a noisy background. Examples include Compact Binary Coalescence (CBC), burst, and glitch detection; see @data_features for a representation of the different features present in gravitational-wave data. *Differentiation* problems, usually known simply as classification problems, involve the separation of detected features into multiple classes, although this is often done in tandem with detection. An example of this kind of problem is glitch classification, in which glitches are classified into classes of known glitch types, and the classifier must separate input data into these classes.

#figure(
    image("data_features.png", width: 100%),
    caption: [ A non-exhaustive hierarchical depiction of some of the features, and proposed features, of gravitational-wave interferometer data. The first fork splits the features into two branches, representing the duration of the features. Here, *continuous* features are defined as features for which it is extremely unlikely for us to witness their start or end within the lifespan of the current gravitational-wave interferometer network and probably the current scientific community @continious_gravitational_waves. These features have durations anywhere from thousands to billions of years. *Transient* features have comparatively short durations @first_detction, from fractions of seconds in the case of stellar-mass Binary Black Hole (BBH) mergers @first_detction to years in the case of supermassive BBH mergers @supermassive_mergers. It should be noted that the detectable period of supermassive BBH binaries could be much longer; although the mergers themselves are transient events, there is no hard cut-off between the long inspiral and merger event. Nevertheless, the mergers are probably frequent enough that some will end within the lifetime of the proposed LISA space probe, so they can be considered transients @supermassive_mergers. The next fork splits features by origin. Features of *astrophysical* origin originate from beyond Earth. This distinction is practically synonymous with the distinction between gravitational waves and signals from other sources since no other astrophysical phenomena are known to have a similar effect in interferometers @first_detction. Features of *terrestrial* origin, unsurprisingly, originate from Earth. These primarily consist of detector glitches caused by seismic activity or experimental artefacts @det_char. Astrophysical transients have a further practical division into CBCs and bursts. The category of *bursts* contains all astrophysical transients that are not CBCs @bursts_O1. The primary reason for this distinction is that CBCs have been detected and have confirmed waveform morphologies. As of the writing of this thesis, no gravitational-wave burst events have been detected @bursts_O1 @bursts_O2 @bursts_O3. Bursts often require different detection techniques; of the proposed sources, many are theorised to have waveforms with a much larger number of free parameters than CBCs, as well as being harder to simulate as the physics are less well-understood @supernovae_waveforms_2 @starquake_detection. These two facts compound to make generating large template banks for such signals extremely difficult. This means that coherence detection techniques that look for coherent patterns across multiple detectors are often used over matched filtering @x-pipeline @oLIB @cWB @BayesWave @MLy. The astrophysical leaves of the diagram represent possible and detected gravitational-wave sources; the text's colourings represent their current status. Green items have been detected using gravitational-wave interferometers, namely the merger of pairs of Binary Black Holes (BBHs) @first_detction, Binary Neutron Stars (BNSs) @first_bns, or one of each (BHNSs) @first_nsbh; see @GWTC-1 @GWTC-2 @GWTC-3 for full catalogues of detections. Yellow items have been detected via gravitational waves but using Pulsar Timing Arrays (PTAs) rather than interferometers @PTA. Blue items represent objects and systems that are theorised to generate gravitational waves and have been detected by electromagnetic observatories but not yet with any form of gravitational wave detection. This includes white dwarf binaries @white_dwarf_binary_em @white_dwarf_lisa_detection, the cosmological background @cosmological_background_em @cosmological_background_gw, starquakes @starquake_em @starquake_gw, and core-collapse supernovae CCSN @supernovae_em @supernovae_gw. This is because they are too weak and/or too uncommon for our current gravitational-wave detector network to have had a chance to detect them. Finally, red items are possible, theorised sources of gravitational waves that have not yet been detected by any means. These are, evidently, the most contentious items presented, and it is very possible that none of these items will ever be detected or exist at all. It should be noted that the number of proposed sources in this final category is extensive, and this is far from an exhaustive list. The presented proposed continious sources are neutron star asymmetries @neutron_star_gw_review, and the presented transient sources are extraterrestrial intelligence @et_gw, cosmic string kinks and cusps @cosmic_string_cusps, accretion disk instabilities @accrection_disk_instability, domain walls @domain_walls, and nonlinear memory effects @non_linear_memory_effects. ]
) <data_features>

@data_features shows that several possible transients with terrestrial and astrophysical origins could be targeted for detection. For our baseline experiments and throughout this thesis, we will select two targets. 

Firstly, *Binary Black Holes (BBHs)*. We have the most numerous detections of BBH signals, and whilst this might make them seem both less interesting and as a solved problem, they have several benefits. As test cases to compare different machine learning techniques against traditional methods, they have the most material for comparison because of their frequency; they would also see the greatest benefits from any computational and speed efficiency savings that may be wrought by the improvement of their detection methods. These factors may become especially relevant when the 3#super[rd] generation detectors, such as the Einstein Telescope and Cosmic Explorer, come online. During their observing periods, they expect detection rates on the order of between $10^4$ and $10^5$ detections per year, which would stretch computing power and cost if current methods remain the only options. In the shorter term, if detection speeds can be improved, faster alerts could be issued to the greater astronomical community, allowing increased opportunity for multimessenger analysis. Only one multimessenger event has thus far been detected --- a Binary Neutron Star (BNS) event, but it is probable, due to the relative similarity in their morphologies, that methods to detect BBHs could be adapted for BNS detection.

Secondly, we will investigate the detection of unmodeled *burst* signals using a machine learning-based coherent detection technique. Bursts are exciting sources whose detection could herald immense opportunities for scientific gain. Possible burst sources include core-collapse supernovae @supernovae_gw, starquakes @starquake_gw, accretion disk instabilities @accrection_disk_instability, nonlinear memory effects @non_linear_memory_effects, domain walls @domain_walls, and cosmic string cusps @cosmic_string_cusps, as well as a plethora of other proposed sources. It should be noted that whilst many bursts have unknown waveform morphologies, some, such as cosmic string cusps, are relatively easy to model and are grouped with bursts primarily due to their as-yet undetected status.

Our current models of the physics of supernovae are limited both by a lack of understanding and computational intractability; detecting the gravitational-wave signal of a supernova could lead to new insights into the supranuclear matter density equation of state as well other macrophysical phenomena present in such events such as neutron transport and hydrodynamics. We may also detect proposed events, such as accretion disk instabilities, which may be missed by standard searches. We can search for the gravitational-wave signals of electromagnetic events which currently have unknown sources, such as fast radio bursts @targeted_frb_search, magnetar flares @targeted_magnetar_search, soft gamma-ray repeaters @targeted_grb_search, and long gamma-ray bursts @targeted_grb_search. Although it's possible that some of these events produce simple, modelable waveforms, it is not currently known, and a general search may one day help to reveal their existence. Some of the more hypothetical proposed sources could fundamentally alter our understanding of the universe, such as evidence for dark matter @domain_wall_dark_matter and/or cosmic strings @cosmic_string_cusps, or if we fail to find them, it could also help to draw limits on theory search space. 

It is unknown whether unmodeled burst detection is a solved problem. Currently, the LIGO-Virgo-KAGRA collaboration has a number of active burst detection pipelines, X-Pipeline @x-pipeline, oLIB @oLIB, Coherent Wave Burst (cWB) @cWB and BayesWave @BayesWave. These include both offline and online searches, including targeted searches wherein a known electromagnetic event is used to limit the search space @targeted_frb_search @targeted_magnetar_search @targeted_grb_search. It could be that the current detection software is adequate and, indeed, the search is hardware rather than software-limited. Even if this is the case, there are probably computational improvements that are possible. It seems unlikely that we have reached the limit of coherent search efficiency.

Traditional coherence techniques require the different detector channels to be aligned for successful detection; therefore, because we don't know a priori the direction of the gravitational-wave sources (unless we are performing a targeted offline search), coherent search pipelines such as X-Pipeline @x-pipeline and cWB @cWB must search over a grid covering all possible incidence directions. In the case of all-sky searches, this grid will necessarily cover the entire celestial sphere. In targeted searches, the grid can be significantly smaller and cover only the uncertainty region of the source that has already been localised by an EM detection. Higher resolution grids will result in a superior search sensitivity; however, they will simultaneously increase computing time. Covering the entire sky with a grid fine enough to achieve the desired sensitivity can be computationally expensive. It is possible to circumnavigate the need to search over a grid using artificial neural networks, shifting much of the computational expense to the training procedure. This has been demonstrated by the MLy pipeline @MLy --- the only fully machine-learning-based pipeline currently in review for hopeful deployment before the end of the fourth observing run (O4). Improvements in the models used for this task could be used to improve the effectiveness of the MLy pipeline. Indeed, some of the work discussed in this thesis was used at an early stage in the pipeline's development to help design the architecture of the models; see @deployment-in-mly. It is hoped that in the future, more aspects of the work shown here can find use in the pipeline's development.

We will focus on the binary detection problem rather than multi-class classification, as there is only one discrete class of BBH (unless you want to draw borders within the BBH parameter space or attempt to discern certain interesting features, such as eccentricity), and in the unmodeled burst case, coherent detection techniques are not usually tuned to particular waveforms, which, in any case, are not widely available for many types of burst. In the next subsection, we will discuss how we can create example datasets to train artificial neural networks for this task.

== Dataset Design and Preparation

In the case of CBCs, we have only a very limited number ($<200$) of example interferometer detections, and in the burst case, we have no confirmed examples. This means that to successfully train artificial neural network models, which typically require datasets with thousands to millions of examples, we must generate a large number of artificial examples. The following subsection describes the creation of these examples, including the acquisition of noise, the generation and scaling of simulated waveforms, and data conditioning.

=== The Power Spectral Density (PSD) <psd-sec>

The Power Spectral Density (PSD) is an important statistical property that is used by several elements of dataset design. Since a custom function was written for this thesis in order to speed up the calculation of the PSD, and since it is helpful to have an understanding of the PSD in order to understand many of the processes described in subsequent sections, a brief explanation is presented.

The PSD is a time-averaged description of the distribution of a time series's power across the frequency spectrum. Unlike a Fourier transform, which provides a one-time snapshot, the PSD conveys an averaged view, accounting for both persistent and transient features; see @psd_eq for a mathematical description. The PSD is used during data conditioning in the whitening transform, wherein the raw interferometer data is processed so that the noise has roughly equal power across the frequency domain, see @feature-eng-sec. For some types of artificial noise generation, the PSD can be used to colour white noise in order to generate more physically active artificial noise; see @noise_acquisition_sec. The PSD is also used to calculate the optimal Signal to Noise ratio, which acts as a metric that can be used to measure the detectability of an obfuscated feature and thus can be used to scale the amplitude of the waveform to a desired detection difficulty.

Imagine a time series composed of a stationary #box("20" + h(1.5pt) + "Hz") sine wave. In the PSD, this would materialise as a distinct peak at #box("20" + h(1.5pt) + "Hz"), effectively capturing the concentrated power at this specific frequency: the frequency is constant, and the energy is localised. If at some time, $t$, we remove the original wave and introduce a new wave at a different frequency, #box("40" + h(1.5pt) + "Hz"), the original peak at #box("20" + h(1.5pt) + "Hz")would attenuate but not vanish, as its power is averaged over the entire time-series duration. Concurrently, a new peak at #box("40" + h(1.5pt) + "Hz") would appear. The power contained in each of the waves, and hence the heights of their respective peaks in the PSD, is determined by the integrated amplitude of their respective oscillations; see @psd-example for a depiction of this example. When applied to a more complicated time series, like interferometer noise, this can be used to generate an easy-to-visualise mapping of the distribution of a time series's power across frequency space.

#show figure: set block(breakable: true) 
#figure(
    image("example_psd.png", width: 100%),
    caption: [Examples of Power Spectral Density (PSD) transforms. _Left:_ Two time domain series. The red series is a #box("20" + h(1.5pt) + "Hz") wave with a duration of #box("0.7" + h(1.5pt) + "s"), and the blue series is this same time series concatenated with a #box("40" + h(1.5pt) + "Hz") wave from $t = 0.7#h(1.5pt)s$ onwards. _Right:_ The two PSDs of the time series are displayed in the left panel. The red PSD was performed across only the #box("0.7" + h(1.5pt) + "s") of the red wave's duration, whereas the blue PSD was taken over the full #box("2.0" + h(1.5pt) + "s") duration. As can be seen, the blue PSD has two peaks, representing the two frequencies of the two waves combined to make the blue time series --- each peak is lower than the red peak, as they are averaged across the full duration, and their respective heights are proportional to their durations as both waves have the same amplitude and vary only in duration.]
) <psd-example>

The PSD can be calculated using Welch's method, which uses a periodogram to calculate the average power in each frequency bin over time. More specifically, the following steps are enacted:

+ First, the time series is split up into $K$ segments of length $L$ samples, with some number of overlapping samples $D$; if $D = 0$, this method is equivalent to Bartlett's method. 
+ Each segment is then windowed with a user-chosen window function, $w(n)$. This is done in order to avoid spectral leakage, avoid discontinuities in the data, smoothly transition between segments, and control several other factors about the method, which allow for fine-tuning to specific requirements.
+ For each windowed segment, $i$, we then estimate the power of the segment, $I_i (f_k)$, at each frequency, $f_k$, by computing the periodogram with

$ I_i (f_k) = 1/L|X_i (k)|^2 $ <periodogram>

where $I_i (f_k)$ is the result of the periodogram, $X_i (k)$ is the FFT of the windowed segment, and $f_k$ is the frequency corresponding to the $k^op("th")$ FFT sample.

4. Finally, we average the periodograms from each segment to get the time-average PSD:

$  S(f_k) =  1/K sum_(i=1)^K I_i (f_k) $ <average_periodograms>

where where $S(f_k)$ is the PSD. Combining @periodogram and @average_periodograms gives

$ S(f_k) =  1/K sum_(i=1)^K 1/L|X_i (k)|^2 $ <psd_eq>

To compute the PSD with enough computational speed to perform rapid whitening and SNR calculation during model training and inference, an existing Welch method from the SciPy scientific Python library @scipy was adapted, converting its use of the NumPy vectorised CPU library @numpy to the TensorFlow GPU library @tensorflow; this converted code is seen in @psd_calculation.

#figure(
```py
@tf.function 
def calculate_psd(
        signal : tf.Tensor,
        nperseg : int,
        noverlap : int = None,
        sample_rate_hertz : float = 1.0,
        mode : str ="mean"
    ) -> (tf.Tensor, tf.Tensor):
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    signal = detrend(signal, axis=-1, type='constant')
    
    # Step 1: Split the signal into overlapping segments
    signal_shape = tf.shape(signal)
    step = nperseg - noverlap
    frames = tf.signal.frame(signal, frame_length=nperseg, frame_step=step)
        
    # Step 2: Apply a window function to each segment
    # Hanning window is used here, but other windows can be applied as well
    window = tf.signal.hann_window(nperseg, dtype = tf.float32)
    windowed_frames = frames * window
    
    # Step 3: Compute the periodogram (scaled, absolute value of FFT) for each 
    # segment
    periodograms = \
        tf.abs(tf.signal.rfft(windowed_frames))**2 / tf.reduce_sum(window**2)
    
    # Step 4: Compute the median or mean of the periodograms based on the 
    #median_mode
    if mode == "median":
        pxx = tfp.stats.percentile(periodograms, 50.0, axis=-2)
    elif mode == "mean":
        pxx = tf.reduce_mean(periodograms, axis=-2)
    else:
        raise "Mode not supported"
    
    # Step 5: Compute the frequencies corresponding to the power spectrum values
    freqs = fftfreq(nperseg, d=1.0/sample_rate_hertz)
    
    #Create mask to multiply all but the 0 and nyquist frequency by 2
    X = pxx.shape[-1]
    mask = \
        tf.concat(
            [
                tf.constant([1.]), 
                tf.ones([X-2], dtype=tf.float32) * 2.0,
                tf.constant([1.])
            ], 
            axis=0
        )
        
    return freqs, (mask*pxx / sample_rate_hertz)


```,
caption : [_Python @python ._ TensorFlow @tensorflow graph function to calculate the PSD of a signal. `signal` is the input time series as a TensorFlow tensor, `nperseg` is the number of samples per segment, $L$, and `noverlap` is the number of overlapping samples, $D$. TensorFlow has been used in order to utilise GPU parallelisation, which offers a significant performance boost over a similar function written in NumPy @numpy.]
) <psd_calculation>

A closely related property, the Amplitude Spectral Density (ASD), is given by the element-wise square root of the Power Spectral Density (PSD)

$ A(f_k) = S(f_k)^(compose 1/2). $ <asd-func>

=== Noise Generation and Acquisition <noise_acquisition_sec>

There are two possible avenues for acquiring background noise to obfuscate our injections. We can either create artificial noise or use real segments extracted from previous observing runs. As was discussed in @interferometer_noise_sec, real interferometer noise is neither Gaussian nor stationary, and many of the noise sources which compose this background are not accounted for or modelled @det_char. This means that any artificial noise will only be an approximation of the real noise --- it is not clear, intuitively, how well this approximation will be suited to training an artificial neural network. 

One perspective argues that using more approximate noise could enhance the network's generalisation capabilities because it prevents overfitting to the specific characteristics of any given noise distribution; this is the approach adopted by the MLy pipeline @MLy. Conversely, another perspective suggests that in order to properly deal with the multitude of complex features present in real noise, we should make our training examples simulate real noise as closely as possible, even suggesting that models should be periodically retrained within the same observing run in order to deal with variations in the noise distribution. These are not discrete philosophies, and the optimal training method could lie somewhere between these two paradigms.

Evidently, in either case, we will want our validation and testing datasets to approximate the desired domain of operation as closely as possible; if they do not, we would have no evidence, other than assumption, that the model would have any practical use in real data analysis. The following subsection will outline the possible types of noise that could be used to create artificial training examples. Throughout the thesis, for all validation purposes, we have used real noise at GPS times, which are not used at any point during the training of models, even when the training has been done on real noise.

*White Gaussian:* The most simplistic and general approach, and therefore probably the most unlike real noise, is to use a white Gaussian background. This is as simplistic as it sounds; we generate $N$ random variables, where N is the number of samples in our noise segment. Each sample is drawn from a normal distribution with a mean of zero and some variance according to your input scaling; often, in the case of machine learning input vectors, this would be unity; see the two uppermost plots in @noise_comparison.

*Coloured Gaussian:* This noise approximation increases the authenticity of the noise distribution by colouring it with a noise spectrum; typically, we use an ASD drawn from the interferometer we are trying to imitate in order to do this; see @psd-sec. By multiplying the frequency domain transform of Gaussian white noise by a given PSD, we can colour that noise with that PSD. The procedure to do this is as follows:

+ Generate white Gaussian noise.
+ Transform the Gaussian noise into the frequency domain using a Real Fast Fourier Transform (RFFT).
+ Multiply the noise frequency spectrum by your selected ASD in order to colour it.
+ Return the newly coloured noise to the time domain by performing an Inverse RFFT (IRFFT).

There are at least two choices of PSD we could use for this process. We could use the PSD of the detector design specification. It represents the optimal PSD given perfect conditions, no unexpected noise sources, and ideal experimental function. This would give a more general, idealistic shape of the PSD across a given observing run. Alternatively, we could use the PSD of a real segment of the background recorded during an observing run; this would contain more anomalies and be a closer approximation to the specific noise during the period for which the PSD was taken. Since the PSD is time-averaged, longer segments will result in more general noise. The MLy pipeline @MLy refers to this latter kind of noise as *pseudo-real* noise; see examples of these noise realisations in the four middle plots of @noise_comparison.

*Real:* Finally, the most authentic type of noise that can be gathered is real interferometer noise. This is noise that has been sampled directly from a detector. Even assuming that you have already decided on which detector you are simulating, which is required for all but white noise generation, there are some extra parameters, shared with the pseudo-real case, that need to be decided. The detector data information, the time period from which you are sampling, and whether to veto any features that may be present in the segment --- e.g. segments which contain events, candidate events, and known glitches. 

To acquire the real data, we utilise the GWPy Python Library's @gwpy data acquisition functionality --- since there are multiple formats in which we could retrieve the data, we must specify some parameters, namely, the frame, the channel, and the state flag. Interferometer output data is stored in a custom file format called a frame file @frame-file; thus, the choice of frame determines the file to be read. Within each frame file lies multiple channels --- each of which contains data from a single output stream. These output streams can be raw data, e.g. raw data from the interferometer photodetector itself; various raw auxiliary data streams, such as from a seismometer; conditioned data, e.g., the primary interferometer output with lines removed; or the state flag channel, which contains information about the status of the detector at every time increment --- the state flag will indicate whether the detector is currently in observing mode or otherwise, so it is important to filter your data for the desired detector state. For the real noise used in this thesis, we use the frame, channel, and state flag, shown in @detector_data_table. We have excluded all events and candidate events listed in the LIGO-Virgo-Kagra (LVK) collaboration event catalogues  @GWTC-1 @GWTC-2 @GWTC-3 but included detector glitches unless otherwise stated.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Detector*],  [*Frame*], [*Channel*], [*State Flag*],
    [LIGO Hanford (H1)], [HOFT_C01], [H1:DCS-CALIB_STRAIN_CLEAN_C01], [DCS-ANALYSIS_READY_C0- 1:1],
    [LIGO Livingston (L1)], [HOFT_C01], [L1:DCS-CALIB_STRAIN_CLEAN_C01], [DCS-ANALYSIS_READY_C0- 1:1],
    [VIRGO (V1)], [V1Online], [V1:Hrec_hoft_16384Hz], [ITF_SCIENCE:1], 
  ),
  caption: [The frame, channel, and state flags used when obtaining data from the respective detectors during the 3#super("rd") observing run (O3). This data was used as obfuscating noise when generating artificial examples to train and validate artificial neural network models throughout this thesis. It should be noted that although the clean channels were produced offline in previous observing runs, the current observing run O4 produces cleaned channels in its online run, so using the cleaned channels during model development ensures that the training, testing, and validation data is closer to what would be the normal operating mode for future detection methods.]
) <detector_data_table>

#figure(
    image("noise_comparison.png", width: 100%),
    caption: [One-second examples of the four possible types of simulated and real noise considered by this thesis. Where real noise is used, it is taken from the LIGO Livingston detector during the third observing run at the GPS times listed. In order, from top to bottom, these are examples of white Gaussian noise, coloured Gaussian noise, pseudo-real noise, and real noise. A description of these noise types and their generation can be found in @noise_acquisition_sec. The left column shows the unaltered values of the noise. Note that the noise has been scaled in all cases except for the pure white noise, which is generated at the correct scale initially. This scaling is used to reduce precision errors and integrate more effectively with the machine learning pipeline, as most loss and activation functions are designed around signal values near unity; see @loss_functions_sec and @activation_functions_sec. The right column shows the same noise realisations after they have been run through a whitening filter. In each case, the PSD of a #box("16.0" + h(1.5pt) + "s") off-source noise segment not displayed is used to generate a Finite Impulse Response (FIR) filter, which is then convolved with the on-source data; see @feature-eng-sec. For the simulated and pseudo-real noise cases, the off-source data is generated using the same method as the on-source data but with a longer duration. In the real noise case, the off-source data consists of real interferometer data drawn from #box("16.5" + h(1.5pt) + "s") before the start of the on-source segment to #box("0.5" + h(1.5pt) + "s") before the start of the on-source segment. This 0.5s gap is introduced because #box("0.5" + h(1.5pt) + "s") must be cropped from the data following the whitening procedure in order to remove edge effects induced via windowing, as well as acting as a buffer to reduce contamination of the off-source data with any features present in the on-source data. Note that the whitened noise plots look very similar for the three simulated noise cases --- a close examination of the data reveals that there is some small variation between the exact values. This similarity occurs because the off-source and on-source noise segments for these examples are generated with identical random seeds and thus have identical underlying noise realisations (which can be seen exactly in the unwhitened white noise plot). Since the PSDs of the on-source and off-source data are nearly identical for the simulated cases, the whitening procedure is almost perfect and reverts it nearly perfectly to its white state. If anything, this similarity boosts confidence that our custom whitening procedure is operating as expected.]
) <noise_comparison>

For our baseline training dataset used in this section, we will employ a real-noise background. An argument can be made that it is an obvious choice. It is the most like real noise, by virtue of being real noise, and thus it contains the full spectrum of noise features that might be present in a real observing run, even if it does not contain the particular peculiarities of any given future observing run in which we may wish to deploy developed models. We will experiment with different noise realisations in a future chapter @noise-type-test-sec.

In each case, we will acquire two seconds of data at a sample rate of #box("2048.0" + h(1.5pt) +"Hz"), which includes #box("0.5" + h(1.5pt) + "s") of data on either side of the time series, which will be cropped after whitening. The whitening is performed similarly in all cases in order to ensure symmetry when comparing obfuscation methods. A power-of-two value is used as it simplifies many of the mathematical operations which need to be performed during signal and injection processing, which may, in some cases, improve performance, as well as help to avoid edge cases that may arise from odd numbers. This frequency was selected as its Nyquist frequency of #box("1024.0" + h(1.5pt) + "Hz") will encompass nearly the entirety of the frequency content of BBH signals; it also covers a large portion of the search space of proposed transient burst sources. The duration of #box("1.0" + h(1.5pt) + "s") is a relatively arbitrary choice; however, it is one which is often the choice for similar examples found in the literature, which makes comparison easier. It also encompasses the majority of the signal power of BBH waves, as well as the theoretically detectable length of many burst sources. For each on-source noise example gathered or generated, #box("16.0" + h(1.5pt) + "s") of off-source background noise is also acquired to use for the whitening procedure; see @feature-eng-sec.

In the case where multiple detectors are being used simultaneously during training or inference, such as coherence detection, noise is generated independently for each interferometer using the same methods, with the restriction that noise acquired from real interferometer data is sampled from each detector within a common time window of #box("2048.0" + h(1.5pt) + "s") so that the noise all originates from a consistent time and date. This is done as there are periodic non-stationary noise features that repeat in daily, weekly, and yearly cycles due to weather, environmental conditions, and human activity. When validating methods, we want to make our validation data as close as possible to reality whilst maintaining the ability to generate large datasets. As we are only ever training our method to operate in real noise conditions (which our validation data attempts to mimic), there is no need to deviate from this method of acquiring noise for our training datasets.

=== Waveform Generation <injection-gen-sec>

Once the background noise has been acquired or generated, the next step is to introduce some differentiation between our two classes, i.e. we need to add a transient signal into some of our noise examples so that our model can find purpose in its existence. When we add a transient into background noise that was not there naturally, we call this an *injection*, since we are artificially injecting a signal into the noise. This injection can be a transient of any type.

Typically, this injection is artificially simulated both due to the limited (or non-existent) number of real examples in many cases and because we will only be able to obtain the real signal through the lens of an interferometer, meaning it will be masked by existing real detector noise. If we were to inject a real injection into some other noise realisation, we would either have to perform a denoising operation (which, even when possible, would add distortion to the true signal) or inject the injection plus existing real noise into the new noise, effectively doubling the present noise and making injection scaling a difficult task. Thus, we will be using simulated injections to generate our training, testing and validation datasets.

Luckily, this is not unprecedented, as most other gravitational-wave detection and parameter estimation methods rely on simulated signals for their operation, including matched filtering. Therefore, there is a well-developed field of research into creating artificial gravitational-wave waveform "approximants", so named because they only approximate real gravitational-wave waveforms. Depending on the complexity and accuracies of the chosen approximant and the source parameter range you are investigating, there will be some level of mismatch between any approximant and the real waveform it is attempting to simulate, even when using state-of-the-art approximants.

To simulate BBH waveforms, we will be using a version of the IMRPhenomD approximant @imrphenom_d, which has been adapted to run on GPUs using NVIDIAs CUDA GPU library. We name this adapted waveform library cuPhenom for consistency with other CUDA libraries such as cuFFT. More details on this adaptation can be found in @software-dev-sec. PhenomD has adjustable parameters which can be altered to generate BBHs across a considerable parameter space, although it should be noted that it does not simulate eccentricity, non-aligned spins, or higher modes. It is also a relatively outdated waveform, with the paper first published in 2015. Newer waveforms, such as those of the IMRPhenomX family, are now available. IMRPhenomD was initially chosen due to its simpler design and as a test case for the adaptation of Phenom approximants to a CUDA implementation. It would not be ideal for implementation into a parameter estimation pipeline due to its mismatch, but the accuracy requirements for a detection pipeline are significantly less stringent.

The IMRPhenomD @imrphenom_d approximant generates a waveform by simulating the Inspiral, Merger, and Ringdown regions of the waveform, hence the IMR in the approximant name. The waveform is generated in the frequency domain before being transformed back into the time domain. The inspiral is generated using post-Newtonian expressions, and the merger ringdown is generated with a phenomenological ansatz; both parts of the model were empirically tuned using a small bank of numerical relativity waveforms. Detailed investigation of aproximant generation was out of the scope of this thesis and will not be covered. See @example_injections for examples of waveforms generated using cuPhenom.

The increased performance of cuPhenom is significant and speeds up the training and iteration process of models considerably. Because of cuPhenom's ability to generate injections on the fly during the training process without significant slowdown, it allows for very quick alteration of dataset parameters for training adjustments. It was felt that this advantage outweighed any gains that would be achieved by using newer waveform models that had not yet been adapted to the GPU, as it seems unlikely, especially in the detection case, that the newer waveform models would make for a significantly harder problem for the model to solve. This statement is, however, only an assumption, and it would be recommended that an investigation is carried out to compare the differences between approximants before any of the methods are used in a real application. A final retraining with these more accurate models would be recommended, in any case.

In the case of unmodelled burst detection, the accuracy of the signal shape is not as fundamental, as the ground truth shapes are not known and, for some proposed events, cover a very large shape space @supernovae-review. In order to cover the entire search space, we have used artificially generated White Noise Bursts (WNBs) generated on the GPU via a simple custom Python @python function utilising TensorFlow @tensorflow. The procedure for generating WNBs with randomised duration and frequency content is as follows.

+ A maximum waveform duration is decided; typically, this would be less or equal to the duration of the example noise that you are injecting the waveform into, with some room for cropping.
+ Arrays of durations, minimum frequencies, and maximum frequencies are generated, each with a number of elements, $N$, equal to the number of waveforms that we wish to generate. These arrays can be pulled from any distribution as long as they follow the following rules. Duration cannot be larger than our maximum requested duration or less than zero. The frequency bounds cannot be less than zero or greater than the Nyquist frequency.
+ It is enforced that the maximum frequency is greater than the minimum frequency for any waveform by swapping values where this is not the case.
+ Gaussian white noise is generated with as many samples, which, given the selected sample rate, will produce a time series with the same duration as our requested max waveform duration.
+ A number of samples at the end of each waveform are zeroed so that each waveform has a number of samples equivalent to the randomised duration assigned to that signal.
+ Each waveform is transformed into the frequency domain by a RFFT. 
+ Samples are zeroed at each end of each frequency-domain signal in order to perform a bandpass and limit the waveform between the assigned frequency constraints for each waveform.
+ The remaining signal is windowed using a Hann window to reduce the effects of the discontinuities generated by the bandpass operation.
+ The frequency domain signal is then returned to the time domain via a IRFFT.
+ Finally, the time-domain waveform is enveloped by a sigmoid window.
+ Assuming the plus polarisation component of the waveform strain was generated first, repeat with the same parameters but different initial noise distributions for the cross polarisation component.

Because we have used random noise across a range of frequency spaces, our distribution will, in theory, cover all possible signals within the specified parameter range. These WNBs can generate waveforms which look qualitatively similar to many proposed burst sources, including current supernovae simulations; see @supernovae_example. See @example_injections for examples of our WNBs and @wnb_calculation for the code used to generate these waveforms.

#figure(
```py 
@tf.function
def generate_white_noise_burst(
    num_waveforms: int,
    sample_rate_hertz: float,
    max_duration_seconds: float,
    duration_seconds: tf.Tensor,
    min_frequency_hertz: tf.Tensor,
    max_frequency_hertz: tf.Tensor
) -> tf.Tensor:
        
    # Casting
    min_frequency_hertz = tf.cast(min_frequency_hertz, tf.float32)
    max_frequency_hertz = tf.cast(max_frequency_hertz, tf.float32)

    # Convert duration to number of samples
    num_samples_array = tf.cast(sample_rate_hertz * duration_seconds, tf.int32)
    max_num_samples = tf.cast(max_duration_seconds * sample_rate_hertz, tf.int32)

    # Generate Gaussian noise
    gaussian_noise = tf.random.normal([num_waveforms, 2, max_num_samples])

    # Create time mask for valid duration
    mask = tf.sequence_mask(num_samples_array, max_num_samples, dtype=tf.float32)
    mask = tf.reverse(mask, axis=[-1])
    mask = tf.expand_dims(mask, axis=1)
    
    # Mask the noise
    white_noise_burst = gaussian_noise * mask

    # Window function
    window = tf.signal.hann_window(max_num_samples)
    windowed_noise = white_noise_burst * window

    # Fourier transform
    noise_freq_domain = tf.signal.rfft(windowed_noise)

    # Frequency index limits
    max_num_samples_f = tf.cast(max_num_samples, tf.float32)
    num_bins = max_num_samples_f // 2 + 1
    nyquist_freq = sample_rate_hertz / 2.0

    min_freq_idx = tf.cast(
        tf.round(min_frequency_hertz * num_bins / nyquist_freq), tf.int32)
    max_freq_idx = tf.cast(
        tf.round(max_frequency_hertz * num_bins / nyquist_freq), tf.int32)

    # Create frequency masks using vectorized operations
    total_freq_bins = max_num_samples // 2 + 1
    freq_indices = tf.range(total_freq_bins, dtype=tf.int32)
    freq_indices = tf.expand_dims(freq_indices, 0)
    min_freq_idx = tf.expand_dims(min_freq_idx, -1)
    max_freq_idx = tf.expand_dims(max_freq_idx, -1)
    lower_mask = freq_indices >= min_freq_idx
    upper_mask = freq_indices <= max_freq_idx
    combined_mask = tf.cast(lower_mask & upper_mask, dtype=tf.complex64)
    combined_mask = tf.expand_dims(combined_mask, axis=1)

    # Filter out undesired frequencies
    filtered_noise_freq = noise_freq_domain * combined_mask

    # Inverse Fourier transform
    filtered_noise = tf.signal.irfft(filtered_noise_freq)
    
    envelopes = generate_envelopes(num_samples_array, max_num_samples)
    envelopes = tf.expand_dims(envelopes, axis=1)
        
    filtered_noise = filtered_noise * envelopes

    return filtered_noise
```,
caption : [_ Python @python . _ TensorFlow @tensorflow graph function to generate the plus and cross polarisations of WNB waveforms; see @injection-gen-sec for a description of the generation method. `num_waveforms` takes an integer value of the number of WNBs we wish to generate. `sample_rate_hertz` defines the sample rate of the data we are working with. `max_duration_seconds` defines the maximum possible duration of any signals within our output data. `duration_seconds`, `min_frequency_hertz`, and `max_frequency_hertz` all accept arrays or in this case TensorFlow tensors, of values with a number of elements equal to `num_waveforms`, each duration. Both polarisations of the WNB are generated with parameters determined by the value of these three arrays at the equivalent index.]
) <wnb_calculation>

#figure(
    image("example_injections.png", width: 100%),
    caption: [Eight simulated waveforms that could be used for injection into noise to form an obfuscated training, testing, or validation example for an artificial neural network. Note that only the plus polarisation component of the strain, $h_plus$, has been plotted in order to increase visual clarity. The leftmost four injections are IMRPhenomD waveforms generated using cuPhenom; see @cuphenom-sec, with parameters (shown in the adjacent grey information boxes) drawn from uniform distributions between #box("5.0" + h(1.5pt) + $M_dot.circle$) and #box("95.0" + h(1.5pt) + $M_dot.circle$) for the mass of both companions and between -0.5 and 0.5 for the dimensionless spin component. Note that during injection generation, the two companions are always reordered so that the mass of companion one is greater and that the IMRPhenomD waveform ignores the x and y spin components. They are included just for code completion. The rightmost four injections consist of WNB waveforms generated via the method described in @injection-gen-sec. Their parameters were again drawn from uniform distributions and are shown in the grey box to their right. The durations were limited between #box("0.1"+ h(1.5pt) + "s") and #box("1.0" + h(1.5pt) + "s"), and the frequencies were limited to between #box("20.0" + h(1.5pt) + "Hz") and #box("500.0" + h(1.5pt) + "Hz"), with the minimum and maximum frequencies automatically swapped.]
) <example_injections>

#figure(
    image("supernova_example.png", width: 80%),
    caption: [The plus polarisation component of the gravitational-wave strain of a simulated core-collapse supernova at a distance of 10 KPc, this data was taken from @supernovae_waveforms. Although some structures can clearly be observed, it is possible to imagine that a method trained to detect WNB signals, such as those presented in @example_injections, might be able to detect the presence of such a signal. ]
) <supernovae_example>

=== Waveform Projection <projection-sec>

As has been discussed, gravitational waves have two polarisation states plus, $plus$, and cross, $times$, which each have their own associated strain values $h_plus$ and $h_times$. Since these strain polarisation states can have different morphologies and since the polarisation angle of an incoming signal paired with a given interferometer's response will alter the proportion of each polarisation that is perceptible by the detector, our aproximant signals are also generated with two polarisation components. Before being injected into any data, the waveforms must be projected onto each detector in our network in order to simulate what that signal would look like when observed with that detector. This projection will account for the full antenna response of each detector. Since a given interferometer has different sensitivities depending on both the direction of the source and the polarisation angle of the incoming wave, some waves will be entirely undetectable in a given detector. 

If we want accurate data when simulating multi-interferometer examples, we must account for both the polarisation angle and direction of the source so that the relative strain amplitudes and morphologies in each detector are physically realistic. 

Since the detectors have a spatial separation, there will usually, depending on source direction, also be a difference in the arrival time of the waves at the different detectors --- this discrepancy is especially important for localising sources, as it provides the possibility for source triangulation, which, along with the antenna responses of each detector, can be used to generate a probability map displaying the probability that a wave originated from a given region of the sky. In coherence detection methods, it also allows for the exclusion of multi-interferometer detections if the detections arise with an arrival time difference greater than that which is physically possible based on the spatial separation of the detectors.

None of this is essential when dealing with single detector examples --- in those cases, we could choose to forgo projection entirely and inject one of the strain polarisation components directly into the obfuscating noise as there are no time separations to model accurately and signal proportionality between detectors is also irrelevant. 

The projection from both the antenna response parameters and the arrival time delay are dependent on the source direction. The plane of the wavefront and the direction of travel of the wave are dependent on the direction of the source. Since the sources are all extremely distant, the wavefront is considered a flat plane. Waves have some time duration, so both the time delay and antenna response parameters will change over the course of the incoming wave's duration as the Earth and the detectors move in space. As we are dealing with relatively short transients ($< 1.0 space s$), the change in these factors will be considered negligible and is not included in projection calculations.

Assuming that we ignore the Earth’s motion, the final waveform present in a detector is given by

$ h(t) = F_plus h_plus (t + Delta t) + F_times h_times (t + Delta t) $ <projection_equ>

where $h(t)$ is the resultant waveform present in the detector output at time $t$; $F_plus$ and $F_times$ are the detector antenna response parameters in the plus and cross polarisations for a given source direction, polarisation angle, and detector; $h_plus$ and $h_times$ are the plus and cross polarisations of the gravitational-wave strain of simulated or real gravitational waves; and $Delta t$ is the arrival time delay taken from a common reference point, often another detector or the Earth’s centre.

We can also calculate the relative times that the signals will arrive at a given detector,

$ Delta t = frac( (accent(x_0, arrow) - accent(x_d, arrow)) ,  c) dot.op accent(m, arrow) $ <time-delay_eq>

where $Delta t$ is the time difference between the wave's arrival at location $accent(x_d, arrow)$ and $accent(x_0, arrow)$, $c$ is the speed of light, $accent(x_0, arrow)$ is some reference location, often taken as the Earth’s centre, $accent(x_d, arrow)$ is the location for which you are calculating the time delay, in our case, one of our interferometers, and $accent(m, arrow)$ is the direction of the gravitational-wave source. If we work in Earth-centred coordinates and take the Earth's centre as the reference position so that $x_0 = [0.0, 0.0, 0.0]$ we can simplify @time-delay_eq to

$ Delta t = - frac(accent(x, arrow) ,  c) dot.op accent(m, arrow). $ <time-delay_eq_sim>

Finally, combining @projection_equ and @time-delay_eq_sim, we arrive at

$ h(t) = F_plus h_plus (t - frac(accent(x, arrow) ,  c) dot.c accent(m, arrow)) + F_times h_times (t - frac(accent(x, arrow) ,  c) dot.c accent(m, arrow)) . $ <final_response_equation>

In practice, for our case of discretely sampled data, we first calculate the effect of the antenna response in each detector and then perform a heterodyne shift to each projection to account for the arrival time differences. When multiple detector outputs are required for training, testing, or validation examples, we will perform these calculations using a GPU-converted version of the PyCBC @pycbc project_wave function; see @projection_examples for example projections.

#figure(
    image("projection_examples.png", width: 80%),
    caption: [Example projection of two artificial gravitational-wave waveforms. The blue waveforms have been projected into the LIGO Livingston interferometer, the red waveforms have been projected into the Ligo Hanford interferometer, and the green waveforms have been projected into the VIRGO interferometer. The left column displays different projections of an IMRPhenomD waveform generated with the cuPhenom GPU library; see @cuphenom-sec. The right column displays different projections of a WNB waveform generated with the method described in @injection-gen-sec. The projections were performed using a GPU adaptation of the PyCBC Python library's @pycbc project_wave function. Both waveforms were projected from different source locations; the projection and time displacement were different in each case. ]
) <projection_examples>

=== Waveform Scaling

Once waveforms have been projected to the correct proportionality, we must have some method to inject them into obfuscating noise with a useful scaling. If using physically scaled approximants, such as the IMRPhenomD waveform, we could forgo scale by calculating the resultant waveform that would be generated by a CBC at a specified distance from Earth, then injecting this into correctly scaled noise (or simply raw real noise). However, since we are also using non-physical waveforms such as WNBs, and because we would like a more convenient method of adjusting the detectability of our waveforms, we will use a method to scale the waveforms to a desired proportionality with the noise.

Evidently, if we injected waveforms that have been scaled to values near unity into real unscaled interferometer noise (which is typically on the order of $10^(-21)$), even a very simple model would not have much of a problem identifying the presence of a feature. Equally, if the reverse were true, no model could see any difference between interferometer data with or without an injection. Thus, we must acquire a method to scale our injections so that their amplitudes have a proportionality with the background noise that is similar to what might be expected from real interferometer data. 

Real data holds a distribution of feature amplitudes, with quieter events appearing in the noise more commonly than louder ones --- this is because gravitational-wave amplitude scales inversely with distance, whereas the volume of searchable space, and thus matter and, generally, the number of systems which can produce gravitational waves, scale cubically with distance from Earth. 

Features with quieter amplitudes will, in general, be harder for a given detection method to identify than features with louder amplitudes. We must design a training dataset that contains a curriculum which maximises model efficacy across our desired regime, with examples that are difficult but never impossible to classify and perhaps some easier cases that can carve channels through the model parameters, which can be used to direct the training of more difficult examples.

In any given noise distribution, there will, for any desired false alarm rate, be a minimum detectable amplitude below which it becomes statistically impossible to make any meaningful detections. This minimum amplitude occurs because even white Gaussian noise will occasionally produce data which looks indistinguishable from a certain amplitude of waveform. 

We can use matched filtering statistics to prove this point, as we know that given an exactly known waveform morphology and perfect Gaussian noise, matched filtering is the optimal detection statistic. The probability that a matched filtering search of one template produces a false alarm is dependant only on the rate at which you are willing to miss true positives. We can use the $cal(F)$-statistic, $cal(F)_0$, for our probability metric to adjust this rate. Assuming that the noise is purely Gaussian, and we are only searching for one specific template, the probability of false detections of this exact waveform, i.e. $P(cal(F) > cal(F)_0)$, can be expressed as

$ P_F (cal(F)_0) = integral_(cal(F)_0)^infinity p_0(cal(F))d cal(F) = exp(-cal(F_0)) sum_(k=0)^(n/2-1) frac(cal(F)_0^k, k!) $ <false_alarm_rate_eq>

where n is the number of degrees of freedom of $chi^2$ distributions. We can see from @false_alarm_rate_eq that the False Alarm Rate (FAR) in this simple matched filtering search is only dependent on your arbitrary choice of $cal(F)_0$. However, in practice, your choice $cal(F)_0$ will be determined by the minimum amplitude waveform you wish to detect because the probability of detection, $P_d$ given the presence of a waveform, is dependent on the optimal SNR of that waveform, $rho$, which has a loose relationship to the amplitude of the waveform. The probability of detection is given by

$ P_D (rho, cal(F)_0) = integral_(cal(F)_0)^infinity frac((2 cal(F))^((n/2 - 1)/2), rho^(n/2 - 1)) I_(n/2-1) (rho sqrt(2 cal(F))) exp(-cal(F) - 1/2 rho^2) d cal(F) $

where $I_(n/2-1)$ is the modified Bessel function of the first kind and order $n/2 -1$. For more information on this, please refer to @gw_gaussian_case.

More complex types of noise, however, like real LIGO interferometer noise, could potentially produce waveform simulacra more often than artificially generated white noise. 

Louder false alarms are less likely than quieter ones, and at a certain amplitude, your detection method will start producing a greater number of false alarms than your desired false alarm rate. If our training dataset includes waveforms with an amplitude that would trigger detections with a false alarm rate near or less than our desired rate, this could significantly reduce the performance of our network, so we must select a minimum amplitude that maximises our detection efficiency at a given false alarm rate. 

Our minimum possible detection amplitude is limited by the combination of the noise and the false alarm rate we desire. There is not a maximum possible signal amplitude, other than some very unuseful upper bound on the closest possible gravitational-wave-producing systems to Earth (a nearby supernova or CBC, for example), but these kinds of upper limit events are so astronomically rare as not to be worth considering. Events will, however, follow a distribution of amplitudes. As is often the case, we can try to generate our training data using a distribution that is as close as possible to the observed data, with the exception of a lower amplitude cutoff, or we can instead use a non-realistic distribution, uniformly or perhaps Gaussianly distributed across some amplitude regime which contains the majority of real signals --- making the assumption that any detection methods we train using this dataset will generalise to higher amplitudes, or failing that, that the missed signals will be so loud that they would not benefit greatly from improved detection methods.

Thus far in this subsection, we have been talking rather nebulously about waveform "amplitude", as if that is an easy thing to define in a signal composed of many continious frequency components. There are at least three properties we might desire from this metric. Firstly, magnitude, some measure of the raw energy contained within the gravitational wave as it passes through Earth --- this measure contains a lot of physical information about the gravitational wave source. Secondly, significance, given the circumstances surrounding the signal, we may want to measure how likely the signal is to have been astrophysical, and finally, closely related to the significance and perhaps most importantly when designing a dataset for artificial neural network training, the detectability, given a chosen detection method this would act as a measure of how easy it is for that method to detect the signal.

Naively, one might assume that simply using the maximum amplitude of the strain, $h_op("peak")$, would be a good measure, and indeed, this would act as a very approximate measure of the ease of detection --- but it is not a complete one. Consider, for a moment, a sine-Gaussian with an extremely short duration on the order of tens of milliseconds but a maximum amplitude that is only slightly louder than a multi-second long BNS signal. You can imagine from this example that the BNS would be considerably easier to detect, but if you were going by $h_op("peak")$ alone, then you would have no idea.

Within gravitational-wave data science, there are nominally two methods for measuring the detectability of a signal --- the root-sum-squared strain amplitude, $h_op("rss")$ and the optimal Signal Noise Ratio (SNR). What follows is a brief description of these metrics.

==== The Root-Sum-Squared strain amplitude, $h_op("rss")$ <hrss-sec>

The Root-Sum-Squared strain amplitude, $h_op("rss")$:, is a fairly simple measure of detectability. Unlike SNR, it is exclusive to gravitational-wave science. It accounts for the power contained across the whole signal by integrating the square of the strain across its duration, essentially finding the area contained by the waveform. It is given by

$ h_op("rss") = sqrt(integral (h_plus (t)^2 + h_times (t)^2 )d t) $

or written in its discrete form, which is more relevant for digital data analysis

$ h_op("rss") = sqrt(sum_(i=1)^(N) (h_plus [t_i]^2 + h_times [t_i]^2)) $

when $h_op("rss")$ is the root-sum-squared strain amplitude, $h_plus (t)$ and $h_times (t)$ are the plus and cross polarisations of the continuous strain, $h_plus (t_i)$ and $h_times (t_i)$ are the plus and cross polarisations of the discrete strain at the i#super("th") data sample, and $N$ is the number of samples in your waveform. 

It should be noted that with any measure that utilises the strain, such as $h_op("peak")$ and $h_op("rss")$, there is some ambiguity concerning where exactly to measure strain. You could, for example, measure the raw strains $h_plus$ and $h_times$ before they have been transformed by the appropriate detector antenna response functions, or you could take the strain $h$ after it has been projected onto a given detector. The advantage of the former is that you can fairly compare the magnitude of different gravitational waves independent of information about the interferometer in which it was detected. This is the commonly accepted definition of the $h_op("rss")$.

The $h_op("rss")$ is most often used during burst analysis as a measure of the detectability, magnitude, and significance of burst transients. Within CBC detection, optimal SNR is often preferred. Whilst $h_op("rss")$ is a simple and convenient measure, it ignores noise, so it cannot by itself tell us if a signal is detectable.

==== Optimal SNR <snr-sec>

The optimal signal-to-noise ratio solves both of these issues by acting as a measure of detectability, magnitude, and significance in comparison to the background noise. Consequently, because it is relative to the noise, the magnitude of a given waveform can only be compared to the SNR of a waveform that was obfuscated by a similar noise distribution. If a real gravitational-wave signal were detected in a single LIGO detector, say, LIGO Hanford, for example, then its SNR would be significantly larger than the same signal detected only in VIRGO, even if the signal was aligned in each case to the original from the optimally detectable sky location. This is because the sensitivity of the VIRGO detector is substantially lower than the two LIGO detectors, so the noise is proportionally louder compared to the waveforms.

It is, however, possibly a good measure of detectability, as detection methods do not much care about the actual magnitude of the signal when they are attempting to analyse one; the only relevant factors, in that case, are the raw data output, consisting of the portion of the gravitational-wave strain perceptible given the detector's antenna response function, see @final_response_equation, and the interferometer noise at that time.

The SNR can also sometimes be an ambiguous measurement, as there are multiple different metrics which are sometimes referred to by this name, most prominently, a ratio between the expected value of the signal and the expected value of the noise, or sometimes the ratio between the root mean square of the signal and noise.  Within gravitational-wave data science, though there is sometimes confusion over the matter, the commonly used definition for SNR is the matched filter SNR. Since matched filtering is the optimal method for detecting a known signal in stationary Gaussian noise, we can use the result of a matched filter of our known signal with that signal plus noise as a measure of the detectability of the signal in a given noise distribution.

The optimal SNR is given by 

$ rho = sqrt(4 integral_0^infinity (|accent(h, tilde.op)(f)|^2)/ S(f) d f) $

where $rho$ is the optimal SNR, S(f) is the one sided PSD, and 

$ accent(h, tilde.op)(f) = integral_(-infinity)^infinity h(x) e^(-i 2 pi f t) d t  $

is the Fourier transform of h(f). The coefficient of 4 is applied since, in order to use only the one-sided transform, we assume that $S(f) = S(-f) $, which is valid because the input time series is entirely real. This applies a factor of two to the output, and since we are only integrating between 0 and $infinity$ rather than $-infinity$ to $infinity$, we apply a further factor of 2. 

Because, again, for data analysis purposes, the discrete calculation is more useful, the optimal SNR of discrete data is given by

$ rho = sqrt(4 sum_(k=1)^(N - 1) (|accent(h, tilde.op)[f_k]|^2)/ S(f_k)) $ <snr-equation>

where $N$ is the number of samples, and, in this case, the discrete Fourier transform $accent(h, tilde)[f]$ is given by

$ accent(h, tilde)[f_k] = sum_(i=1)^(N - 1) h[t_i] e^(-(2 pi)/N k i)  $ <fourier-transform-eq>

For the work during this thesis, we have used a TensorFlow @tensorflow implementation for calculating the SNR of a measure. This implementation is show in @snr_calculation.

#figure(
```py 
@tf.function 
def calculate_snr(
    injection: tf.Tensor, 
    background: tf.Tensor,
    sample_rate_hertz: float, 
    fft_duration_seconds: float = 4.0, 
    overlap_duration_seconds: float = 2.0,
    lower_frequency_cutoff: float = 20.0,
    ) -> tf.Tensor:
    
    injection_num_samples      = injection.shape[-1]
    injection_duration_seconds = injection_num_samples / sample_rate_hertz
        
    # Check if input is 1D or 2D
    is_1d = len(injection.shape) == 1
    if is_1d:
        # If 1D, add an extra dimension
        injection = tf.expand_dims(injection, axis=0)
        background = tf.expand_dims(background, axis=0)
        
    overlap_num_samples = int(sample_rate_hertz*overlap_duration_seconds)
    fft_num_samples     = int(sample_rate_hertz*fft_duration_seconds)
    
    # Set the frequency integration limits
    upper_frequency_cutoff = int(sample_rate_hertz / 2.0)

    # Calculate and normalize the Fourier transform of the signal
    inj_fft = tf.signal.rfft(injection) / sample_rate_hertz
    df = 1.0 / injection_duration_seconds
    fsamples = \
        tf.range(0, (injection_num_samples // 2 + 1), dtype=tf.float32) * df

    # Get rid of DC
    inj_fft_no_dc  = inj_fft[:,1:]
    fsamples_no_dc = fsamples[1:]

    # Calculate PSD of the background noise
    freqs, psd = \
        calculate_psd(
            background, 
            sample_rate_hertz = sample_rate_hertz, 
            nperseg           = fft_num_samples, 
            noverlap          = overlap_num_samples,
            mode="mean"
        )
            
    # Interpolate ASD to match the length of the original signal    
    freqs = tf.cast(freqs, tf.float32)
    psd_interp = \
        tfp.math.interp_regular_1d_grid(
            fsamples_no_dc, freqs[0], freqs[-1], psd, axis=-1
        )
        
    # Compute the frequency window for SNR calculation
    start_freq_num_samples = \
        find_closest(fsamples_no_dc, lower_frequency_cutoff)
    end_freq_num_samples = \
        find_closest(fsamples_no_dc, upper_frequency_cutoff)
    
    # Compute the SNR numerator in the frequency window
    inj_fft_squared = tf.abs(inj_fft_no_dc*tf.math.conj(inj_fft_no_dc))    
    
    snr_numerator = \
        inj_fft_squared[:,start_freq_num_samples:end_freq_num_samples]
    
    if len(injection.shape) == 2:
        # Use the interpolated ASD in the frequency window for SNR calculation
        snr_denominator = psd_interp[:,start_freq_num_samples:end_freq_num_samples]
    elif len(injection.shape) == 3: 
        snr_denominator = psd_interp[:, :, start_freq_num_samples:end_freq_num_samples]
        
    # Calculate the SNR
    SNR = tf.math.sqrt(
        (4.0 / injection_duration_seconds) 
        * tf.reduce_sum(snr_numerator / snr_denominator, axis = -1)
    )
    
    SNR = tf.where(tf.math.is_inf(SNR), 0.0, SNR)
    
    # If input was 1D, return 1D
    if is_1d:
        SNR = SNR[0]

    return SNR
```,
caption : [_ Python @python. _ TensorFlow @tensorflow graph function to calculate the SNR of a signal. `injection` is the input signal as a TensorFlow tensor, `background` is the noise into which the waveform is being injected, `sample_rate_hertz` is the sample rate of both the signal and the background, `fft_duration_seconds` is the duration of the FFT window used in the PSD calculation, `overlap_duration_seconds` is the duration of the overlap of the FFT window in the PSD calculation, and `lower_frequency_cutoff` is the frequency of the lowpass filter, below which the frequency elements are silenced.]
) <snr_calculation>

Once the optimal SNR or $h_op("rss")$ of an injection has been calculated, it is trivial to scale that injection to any desired optimal SNR or $h_op("rss")$ value. Since both metrics scale linearly when the same coefficient scales each sample in the injection,

$ h_op("scaled") = h_op("unscaled") M_op("desired") / M_op("current") $ <scaling-equation>

where $h_op("scaled")$ is the injection strain after scaling, $h_op("unscaling")$ is the injection strain before scaling, $M_op("desired")$ is the desired metric value, e.g. $h_op("rss")$, or SNR, and $M_op("current")$ is the current metric value, again either $h_op("rss")$, or SNR. Note that since $h_op("rss")$ and SNR are calculated using different representations of the strain, $h_op("rss")$ before projection into a detector, and SNR after, the order of operations will be different depending on the scaling metric of choice, ie. for $h_op("rss")$: scale $arrow$ project, and for SNR: project $arrow$ scale. 

#figure(
    image("scaling_comparison.png", width: 100%),
    caption: [Eight examples of artificial injections scaled to a particular scaling metric and added to a real noise background to show variance between different scaling methods. The blue line demonstrates the whitened background noise plus injection; the red line represents the injection after being run through the same whitening transform as the noise plus injection, and the green line represents the injection after scaling to the desired metric. The leftmost column contains an IMRPhenomD waveform, generated using cuPhenom, injected into a selection of various background noise segments and scaled using SNR; see @snr-sec. From upper to lower, the SNR values are 4, 8, 12, and 16, respectively. The rightmost column displays a WNB injected into various noise distributions, this time scaled using $h_op("rss")$; see @hrss-sec. From upper to lower, the $h_op("rss")$ values are as follows: $8.52 times 10^(-22)$, $1.70 times 10^(-21)$, $2.55 times 10^(-21)$, and $3.41 times 10^(-21)$. As can be seen, though both sequences are increasing in linear steps with a uniform spacing of their respective metrics, they do not keep in step with each other, meaning that if we double the optimal SNR of a signal, the $h_op("rss")$ does not necessarily also double.]
) <scaling_comparison>

For the experiments performed later in this section, we will use SNR as our scaling metric drawn from a uniform distribution with a lower cutoff of 8 and an upper cutoff of 20. These values are rough estimates of a desirable distribution given the SNR values of previous CBC detections.

If we wish to utilise multiple detectors simultaneously as our model input, we can scale the injections using either the network SNR or the $h_op("rss")$ before projection into the detectors. In the case of $h_op("rss")$, the scaling method is identical, performed before detection and injection. Network SNR is computed by summing individual detector SNRs in quadrature, as shown by

$ rho_op("network") = sqrt(sum_(i=1)^(N) rho_i^2) $ <network-snr>

where $rho_op("network")$ is the network SNR, $N$ is the total number of detectors included in the input, and $rho_i$ is the detector SNR of the i#super("th") detector given in each case by @snr-equation. To scale to the network, SNR @scaling-equation can still be used, with the network SNR of @network-snr as the scaling metric, by multiplying the resultant projected injection in each detector by the scaling coefficient.

=== Data Dimensionality and Layout <dim_sec>

Interferometer output data is reasonably different from the example MNIST data we have been using to train models thus far, the primary difference being that it is one-dimensional rather than two, being more similar to audio than image data. In fact, most of the features we are looking for within the data have a frequency that, when converted to sound, would be audible to the human ear, so it is often useful to think of the problem in terms of audio classification. In many ways, this reduced dimensionality is a simplification of the image case. In pure dense networks, for example, we no longer have to flatten the data before feeding it into the model.

There are, however, multiple interferometers across the world. During an observing run, at any given time, there are anywhere between zero to five operational detectors online: LIGO Livingston (L1), LIGO Hanford (H1), Virgo (V1), Kagra (K1), and GEO600 (G1). GEO600 is not considered sensitive enough to detect any signals other than ones that would have to be so local as to be rare enough to dismiss the probability, so it is usually not considered for such analysis. It should also be noted that during O4, both Virgo and Kagra are currently operating with a sensitivity and up-time frequency that makes it unlikely they will be of assistance for any detection. It is hoped that the situation at these detectors will improve for future observing runs. As such, it is possible to include multiple detectors within our model input, and in fact, such a thing is necessary for coherence detection to be possible.

This multiplicity brings some complications in the construction of the input examples. Currently, we have only seen models that ignore the input dimensionality; however, with other network architectures, such as Convolutional Neural Networks (CNNs), this is not always the case. Therefore, we must consider the data layout. In the simplest cases, where we are not modifying the shape of the data before injection, we can imagine three ways to arrange the arrays; see @layout_options for a visual representation.

- *Lengthwise*: wherein the multiple detectors are concatenated end to end, increasing the length of the input array by a factor equal to the number of detectors. This would evidently still be a 1D problem, just an extended one. While perhaps this is the simplest treatment, we can imagine that this might perhaps be the hardest to interpret by the model, as we are mostly discarding the dimensionality, although no information is technically lost.
- *Depthwise*: Here, the detectors are stacked in the depth dimension, an extra dimension that is not counted toward the dimensionality of the problem, as it is a required axis for the implementation of CNNs, in which each slice represents a different feature map; see @cnn_sec. Often, this is how colour images are injected by CNNs, with the red, green, and blue channels each taking up a feature map. This would seem an appropriate arrangement for the detectors. However, there is one significant difference between the case of the three-colour image and the stacked detectors, that being the difference in signal arrival time between detectors; this means that the signal will be offset in each channel. It is not intuitively clear how this will affect model performance, so this will have to be empirically compared to the other two layouts.
- *Heightwise*: The last possible data layout that could be envisioned is to increase the problem from a 1D problem to a 2D one. By concatenating the arrays along their height dimension, the 1D array can be increased to a 2D array.

#figure(
    image("data_layout.png", width: 100%),
    caption: [Possible data layouts for multi-detector examples. Here, $d$ is the number of included detectors, and $N$ is the number of input elements per time series. There are three possible ways to align interferometer time-series data from multiple detectors. These layouts are discussed in more detail in @dim_sec. ]
) <layout_options>

For pattern-matching methods, like that which is possible in the CBC case, there are also advantages to treating each detector independently. If we do this, we can use the results from each model as independent statistics, which can then be combined to create a result with a far superior False Alarm Rate (FAR). We could combine the score from both models and calculate a false alarm rate empirically using this combined score, or use each detector as a boolean output indicating the presence of a detector or not, and combine the FARs using @comb_far_eq.

For the first case treating the two models as one, the combined score is calculated by

$ op("S")_(op("comb")) = product_(i=1)^N op("S")_i $ 

where $op("S")_(op("comb"))$ is the combined classification score, which can be treated approximately as a probability if the output layer uses a softmax, or single sigmoid, activation function, see @softmax-sec, $op("S")_i$ is the output score of the $i^op("th")$ classifier input with the data from the $i^op("th")$ detector, and $N$ is the number of included detectors. Note that one could employ a uniquely trained and/or designed model for each detector or use the same model for each detector.

In the second case, treating each model as an independent boolean statistic and assuming that the output of the detectors is entirely independent except for any potential signal, the equation for combining FARs is

$ op("FAR")_(op("comb")) = (w - o)  product_(i=1)^N op("FAR")_i $ <comb_far_eq>

where $op("FAR")_(op("comb"))$ is the combined FAR, $N$ is the number of included detectors, $w$ is the duration of the input vector, and $o$ is the overlap between windows. This equation works in the case when your detection method tells you a feature has been detected within a certain time window, $w$, but not the specific time during that window, meaning that $t_"central" > w_"start" and t_"central" < w_"end"$, where $t_"central"$ is the signal central time, $w_"start"$ is the input vector start time and $w_"end"$ is the input vector end time. 

If your detection method can be used to ascertain a more constrained time for your feature ($w_"duration" < "light_travel_time"$), then you can use the light travel time between the two detectors to calculate a FAR. For two detectors, combing the FAR in this way can be achieved by

$ op("FAR")_(1, 2) = 2 op("FAR")_1 op("FAR")_2 w_(1,2) $

where $op("FAR")_(op("comb"))$ is the combined FAR, and $w_(1,2)$ is the light travel time between detectors 1 and 2, as this is the largest physically possible signal arrival time separation between detectors; gravitational waves travel at the speed of light, and detector arrival time difference is maximised if the direction of travel of the wave is parallel to the straight-line path between the two detectors.
/*
Generalising this kind of FAR combination to $N$ detectors becomes difficult because each pair of detectors will have a different maximum time separation, meaning that the sequence in which the detection occurs becomes important. The general formula is

$ op("FAR")_(op("comb")) = sum_op("all sequences S") op("FAR")_(S_1) times product_(i=1)^(N-1) op("FAR")_(S_(K+1)) times w_(S_(K) S_(K+1)) $

where a given sequence S is an order of detectors, i.e. $S = [1,2,3...N]$, and $"all sequences S"$ are all possible combinations of $S$, meaning that there are $N!$ possible sequences. For a more simplistic approach that will slightly overestimate the true FAR value, the maximum time separation between any pair of detectors is sometimes used instead of unique separations for each pair. Note that Equation 59 is the general case where only cases when all $N$ detectors have detections are considered detections; if less than $N$ detectors are required for detection, then we should use all combinations of sequences of length $X$ when $X$ is the number of detectors required for a confirmed detection.

*/

In the case where we are using $t_"central"$ and coincidence times to calculate our combined FAR, if we use overlapping data segments to feed our model, we must first group detections which appear in multiple inferences and find one central time for the detection. We can use an empirical method to determine how best to perform this grouping and identify if and how model sensitivity varies across the input window.

=== Feature Engineering and Data Conditioning <feature-eng-sec>

Invariably, there are data transforms that could be performed prior to ingestion by the model. If there are operations that we imagine might make the task at hand easier for the model, we can perform these transforms to improve network performance. Because we are attempting to present the data to the model in a form that makes the features easier to extract, this method of prior data conditioning is known as *feature engineering*. It should be noted that feature engineering does not necessarily add any extra information to the data. In fact, in many cases, it can reduce the overall information content whilst simultaneously simplifying the function that the model is required to approximate in order to operate as intended, see for example the whitening proceedure descibed in @whitening-sec. As we have said before, although the dense neural network is, at its limit, a universal function approximator, there are practical limitations to finding the right architecture and parameters for a given function, so sometimes simplifying the task can be beneficial. This can reduce the model size and training time, as well as improve achievable model performance when the time available for model and training optimisation is limited.

==== Raw Data

When designing the package of information that will be presented to the network at each inference, the simplest approach would be to feed the raw interferometer data directly into the model. There are certainly some methodologies that consider it optimal to present your model with as much unaltered information as possible. By performing little to no data conditioning, you are allowing the network to find the optimal path to its solution; if all the information is present and an adequate model architecture is instantiated, then a model should be able to approximate the majority of possible conditioning transforms during model training, not only this, but it may be able to find more optimal solutions that you have not thought of, perhaps ones customised to the specific problem at hand, rather than the more general solutions that a human architect is likely to employ. This methodology, however, assumes that you can find this adequate model architecture and have an adequate training procedure and dataset to reach the same endpoint that could be achieved by conditioning your data. This could be a more difficult task than achieving a result that is almost as good with the use of feature engineering.

==== Whitened Data <whitening-sec>

One type of data conditioning that we will employ is time-series whitening. As we have seen in @interferometer_noise_sec, as well as containing transient glitches, the interferometer background is composed of many different continuous quasi-stationary sources of noise, the frequency distributions of which compose a background that is unevenly distributed across our frequency search space. This leaves us with 1D time series that have noise frequency components with much greater power than any interesting features hidden within the data. This could potentially make detections using most methods, including artificial neural networks, much more difficult, especially when working in the time domain; see @whitnening_examples for an example of the PSD of unwhitened noise.

#figure(
    image("whitening_examples.png", width: 100%),
    caption: [An example of a segment of interferometer data before and after whitening. The two leftmost plots in blue show the PSD, _upper_, and raw data, _lower_, output from the LIGO Hanford detector before any whitening procedure was performed. The two rightmost plots show the same data after the whitening procedure described in @whitening-sec has been implemented. The data was whitened using the ASD of a #box("16.0" + h(1.5pt) + "s") off-source window from #box("16.5" + h(1.5pt) + "s") before the start of the on-source window to #box("0.5" + h(1.5pt) +"s") before. The #box("0.5" + h(1.5pt) +"s") gap is introduced as some data must be cropped after whitening due to edge effects caused by windowing. This also acts to ensure that it is less likely that any features in the on-source data contaminate the off-source data, which helps reduce the chance that we inadvertently whiten any interesting features out of the data.]
) <whitnening_examples>

Fortunately, there exists a method to flatten the noise spectrum of a given time series whilst minimising the loss of any transient features that don't exist in the noise spectrum. This requires an estimate of the noise spectrum of the time series in question, which does not contain the hidden feature. In this case, this noise spectrum will take the form of an ASD; see @asd-func. 

Since the noise spectrum of the interferometer varies with time, a period of noise close to but not overlapping with the section of detector data selected for analysis must be chosen --- we call this time series the *off-source* period. The period being analysed, the *on-source* period, is not included in the off-source period so that any potential hidden features that are being searched for, e.g. a CBC signal, do not contribute significant frequency components to the ASD, which may otherwise end up dampening the signal along with the noise during the whitening procedure. It should be noted, then, that whitening via this process uses additional information from the off-source period that is not present in the on-source data. During this thesis, we have elected to use an off-source window duration of #box("16.0" + h(1.5pt) +"s"), as this was found to be an optimal duration by experiments performed by as past of previous work during the development of MLy @MLy, although it should be noted that we have taken the on-source and crop regions after the off-source as opposed to the initial MLy experiments wherein it was taken at the centre of the off-source window. See @onsource_offsource_regions for a depiction of the relative locations of the on-source and off-source segments.

#figure(
    image("onsource_offsource_regions.png", width: 100%),
    caption: [Demostration of the on-source and off-source regions used to calculate the ASD used during the whitening operations throughout this thesis wherever real noise is utilised. Where artificial noise is used, the off-source and on-source segments are generated independently but with durations equivalent to what is displayed above. The blue region shows the #box("16.0" + h(1.5pt) + "s") off-source period, the green region shows the #box("1.0" + h(1.5pt) + "s") on-source period, and the two red regions represent the #box("0.5" + h(1.5pt) + "s") crop periods, which are removed after whitening. During an online search, the on-source region would advance in second-long steps, or if some overlap was implemented, less than second-long steps, meaning all data would eventually be searched. The leading #box("0.5" + h(1.5pt) + "s") crop region will introduce an extra #box("0.5" + h(1.5pt) + "s") of latency to any search pipeline. It may be possible to avoid this latency with alternate whitening methods, but that has not been discussed here. ]
) <onsource_offsource_regions>

We can whiten the data by convolving it with a suitably designed Finite Impulse Response (FIR) filter. This procedure is described by the following steps:

+ Calculate the ASD using @asd-func, this will act as the transfer function, $G(f)$, for generating the FIR filter. This transfer function is a measure of the frequency response of the noise in our system, and during the whitening process, we will essentially try to normalise the on-source by this off-source noise in order to flatten its PSD. We generate a filter with a #box("1" + h(1.5pt) + "s") duration.
+ Next, we zero out the low and high-frequency edges of the transfer function with

$ G_"trunc" (f) = cases(
  0 "if" f <= f_"corner",
  G(f) "if" f_"corner" < f < f_"Nyquist" - f_"corner",
  0 "if" >= f_"Nyquist" - f_"corner"  
). $

 This stage discards frequency components which we no longer care about both because these frequencies are outside of the band we are most interested in and because discarding them can improve function stability and performance whilst reducing artifacting.
 
3. Optionally, we can apply a Planc-taper window to smooth the discontinuities generated by step 2; we will apply this window in all cases. The Planc-taper window has a flat centre with smoothly tapering edges, thus the windowing is only applied as such to remove discontinuites whilst affecting the central region as little as possible.

$ G_"smoothed" (f) = G_"trunc" (f) dot W(f). $

4. Next we compute the inverse Fourier transform of $G_"smoothed" (f)$ to get the FIR filter, $g(t)$, with 

$ g(t) = 1 / (2 pi) integral_(-infinity)^infinity G_"smoothed" (f) e^(j f t) d f. $

This creates a time-domain representation of our noise characteristics, which can then be used as a filter to remove similar noise from another time-domain signal. In practice, we utilise an RFFT function to perform this operation on discrete data. As opposed to an FFT, this transform utilises symmetries inherent when transforming from complex to real data in order to halve the computational and memory requirements.

5. Finally, we convolve our FIR filter, $g(t)$, with the data we wish to whiten, $x(t)$,

$ x_"whitened" (t) = x(t) ast.op g(t) $ 

where $x_"whitened" (t)$ is the resultant whitened time-series, $x(t)$ is the original unwhitened data, and $g(t)$ is the FIR filter generated from the off-source ASD. This convolution effectively divides the power of the noise at each frequency by the corresponding value in $G(f)$. This flattens the PSD, making the noise uniform across frequencies; see @whitnening_examples for an example of this transform being applied to real interferometer data.

This method was adapted from the GWPy Python library @gwpy and converted from using NumPy functions @numpy to TensorFlow GPU operations @tensorflow in order to work in tandem with the rest of the training pipeline and allow for rapid whitening during the training process.

==== Pearson Correlation

A method of feature engineering that is employed prominently by the MLy pipeline @MLy involves extracting cross-detector correlation using the Pearson correlation. The Pearson correlation is given by

$ r = frac( N (sum_(i=0)^N x_i y_i) - (sum_(i=0)^N x_i) (sum_(i=0)^N y_i ) , sqrt( [N sum_(i=0)^N x_i^2 - (sum_(i=0)^N x_i)^2] times [N sum_(i=0)^N y_i^2 - (sum_(i=0)^N y_i)^2] ) ) $

where r is the Pearson correlation coefficient, N is the number of data points in each input array, and $x_i$ and $y_i$ are the i#super("th") elements of the $accent(x, arrow)$ and $accent(y, arrow)$ arrays respectively.

Nominally, this produces one scalar output value given two input vectors, $accent(x, arrow)$ and $accent(y, arrow)$, of equal length, $N$. A value of $r = 1$ indicates perfect correlation between the two vectors, whereas a value of $r = -1$ indicates perfect anti-correlation. Finally, a value of $r = 0$ indicates no correlation between the vectors. Note that if one of the vectors is entirely uniform, then the result is undefined. 

This calculation assumes that the two vectors are aligned such that the value in $x_i$ corresponds to the value in $y_i$. If this is not the case, as would happen for interferometer data if there is an arrival time difference (which there will be for most sky locations), then this will be an imperfect measure of correlation, even discarding the obfuscation of the noise. Because, as was discussed previously in @projection-sec, we do not know the direction of the source a priori, MLy @MLy calculates the correlation for all possible arrival times given the light travel time between the two detectors in question. It uses minimum increments of the sample duration so that no heterodyning is necessary. This is done with the assumption that any difference in arrival time less than the sample duration will have a negligible effect on the correlation. It should be noted that this method is still hampered by the different polarisation projections dependent on the source polarization and by the obfuscating noise. See @pearson_example for examples of the rolling Pearson correlation calculated for LIGO Hanford and LIGO Livingston interferometer data.

#figure(
    image("pearson_example.png", width: 100%),
    caption: [Example whitened on-source and correlation plots of real interferometer noise from a pair of detectors, in this case, LIGO Livingston and LIGO Hanford, with either coherent, incoherent, or no injections added. The leftmost plots adjacent to the info panels are grouped into pairs. In each case, LIGO Livingston is at the top, and LIGO Hanford is underneath. Identical on-source and off-source noise segments were used for each example of the same detector, and noise for each detector was gathered with a time difference of no more than #box("2048.0" + h(1.5pt) + "s"). In the leftmost plots, the green series is the unwhitened but projected waveform to be injected into the real noise from that detector. The red series is that same injection but subject to the same whitening procedure that will also be applied to the on-source plus injections, and the blue series is the whitened on-source plus injections. The rightmost plots each correspond to a pair of detectors and display the rolling Pearson correlation values between those two whitened on-source plus injection series. Since there is approximately a max arrival time difference of #box("0.01" + h(1.5pt) + "s") between LIGO Livingston and LIGO Hanford, the number of correlation calculations performed corresponds to the rounded number of samples required to represent #box("0.02" + h(1.5pt) + "s") of data at #box("2048.0" + h(1.5pt) + "Hz"). This number is two times the maximum arrival time difference because the difference could be positive or negative. In this case, that difference comes to 40 samples. All injections have been scaled to an optimal network SNR of 30 using the method described in @snr-sec. The upper pair of detectors has no injection. As would be expected, the correlation is low regardless of the assumed arrival time difference. The second pair from the top has been injected with a coherent white noise burst (WNB), see @injection-gen-sec, which has been projected onto the two detectors using a physically realistic mechanism previously described in @projection-sec. Here, the correlation is much stronger. We can see it rise and fall as the waveforms come in and out of coherence. The third from the top, the central plot, shows an injection of two incoherent WNBs. They are processed identically to the coherent case, but the initial waveforms are generated independently, including their durations. The Pearson correlation looks very similar to the pure noise case in the uppermost plot, as might be expected. The second from the lowest pair has been injected with a coherent IMRPhenomD waveform, which again has been correctly projected. We can observe that a small correlation is observed at an arrival time difference of around #box("0.005" + h(1.5pt) + "s"), suggesting that the two waveforms arrived at the detectors #box("0.005" + h(1.5pt) + "s") apart. Finally, the lowest plot depicts two incoherent IMRPhenomD waveforms projected into the noise. Though these are generated with different parameters, the shared similarities in morphology between all CBC waveforms cause correlation to be registered. By maximum amplitude alone, it may even appear as though there is more correlation happening here than in the correlated case. This highlights one potential weakness of using the Pearson correlation, which can sometimes show some degree of correlation even if the two waveforms were not produced using the same physically simulated mechanism.]
) <pearson_example>

As with most mathematical functions, we have created a new GPU-based function for the calculation of the Pearson correlation in Python @python, using the TensorFlow GPU library @tensorflow, for computational speed and easy integration with the rest of the pipeline.

==== Fourier Transform

So far, we have looked at data conditioning, which produces results in the time domain. As we know, and as has been demonstrated by the previous discussion, many aspects of time series processing are performed in the frequency domain. Often, features which are hard to distinguish in the time domain are relatively easy to spot in the frequency domain, even with the human eye. Many have characteristic morphologies, such as distinct lines due to powerline harmonics and violin modes. If we make the assumption that if it is easier for a human, it might also be easier for a machine learning method, we should certainly examine feature engineering methods that take us into the frequency domain. The most obvious way to do this would be to use a simple Fourier transform, which takes us directly from a time-domain series to a frequency-domain one. The discrete form of the Fourier transform is given above in @fourier-transform-eq.

==== Power Spectral Density (PSD) and Amplitude Spectral Density (ASD)

As discussed in @psd-sec, the PSD is used in many calculations and transforms in gravitational wave data analysis, so it makes sense that along with the closely related property, the ASD, it may also be useful information to provide to a model. Since the PSD has already been discussed in detail in @psd-sec, we will not linger on it here.

==== Spectrograms

The final feature engineering method that we will discuss allows us to represent data in both the time and frequency domains simultaneously. Spectrograms are visualisations of the Short-Time Fourier Transform (STFT) of a time series. The STFT is computed by dividing a time series into many smaller periods, much like in the calculation of a PSD; however, instead of being averaged, you can simply use this 2D output as an image in its own right, which displays how the frequency components of a time series fluctuate over its duration. This retains some information from the time domain. The 2D STFT of a continuous time series, $x(t)$, is given by

$ op("STFT")(x)(t, f) = integral_(-infinity)^infinity x(tau) w(t - tau) e^(-i 2 pi f tau) d tau  $

where $op("STFT")(x)(f, t)$ is the value of the STFT of $x(t)$ at a given time, $t$, and frequency, $f$, $w(t)$ is a configurable window function that helps to minimize the boundary effects, and $tau$ is a dummy integration variable used to navigate through the time domain at the expense of loosing some information from the frequency domain, making the spectrogram, like whitening, a lossy transform. In its discrete form, this becomes

$  op("STFT")(x)[n, k] = sum_(m = 0)^(N-1) x[m] w[n - m] e^((-i 2 pi k m) / N)  $ <stft-eq>

where $op("STFT")(x)[n, k]$ is the value of the discrete STFT of a discrete time series, $x[m]$ at a given time index, $n$, and frequency index, $k$, $w[t]$ is a discrete window function, and N is the number of samples in our discrete time series. It should be noted that there are two time indices present, $n$ and $m$, because a reduction in dimensionality along the time axis usually occurs since the step between adjacent FFT segments is commonly greater than one.

When creating a spectrogram, the values are typically squared,

$ S[k, n] = (op("STFT")(x)[n, k])^2 $ <stft_sq>

to represent the power of the frequency components, similar to the process of calculating the PSD. Alternatively, the magnitude can be taken with

$ S[k, n] = |op("STFT")(x)[n, k]|. $

Before plotting, the data is often converted into decibels to better visualise the dynamic range,

$ op("DATA") = 10 times log (S[k, n]). $ <dec-eq>

We have created a custom Python TensorFlow function @tensorflow to perform these calculations on the GPU; see @spectrogram_examples for illustrations of this in use on real noise with injected waveform approximants. As is the case with multiple 1D time series, the question also remains of how to combine multiple spectrograms in the case of multiple detector outputs, see @dim_sec.

#figure(
    image("spectrogram_examples.png", width: 85%),
    caption: [Six example noise segments and their corresponding spectrograms. In all cases, the noise is real interferometer data acquired from the LIGO Hanford detector during the 3#super("rd") observing run. It is whitened using the procedure described in @whitening-sec. For the time series plots, the green series represents the original, unwhitened waveform before injection, the red series is the waveform with the same whitening transform applied to it as was applied to the on-source background plus injection, and the blue series is the whitened on-source background plus injection, except for the first two time series plots which contain no injection. The spectrograms were generated using the STFT described by @stft-eq, converted into power with @stft_sq, and finally transformed into a decibel logarithmic scale for plotting using @dec-eq. The two uppermost plots and their respective spectrograms have no injections. The two middle plots and their respective spectrograms have IMRPhenomD @imrphenom_d approximants created with cuPhenom injected into the noise, and the two lower plots and their respective spectrograms, have White Noise Burst (WNB) waveforms generated using the method described in @injection-gen-sec, injected into the noise. In all cases, the injections were scaled to an optimal SNR randomly selected between 15 and 30; these are quite high values chosen to emphasise the features in the spectrograms. As can be seen, the whitened noise that contains injected features has spectrograms with highlighted frequency bins that have a magnitude much larger than the surrounding background noise; the different signal morphologies also create very different shapes in the spectrograms. This allows us to see the frequency components of the signal more easily, to observe the presence of interesting features, and differentiate between the WNB and the CBC case. ]
) <spectrogram_examples>

==== Summary

There are multiple different possibilities for how to condition the data before it is fed into any potential machine learning model; see @feature-enginering-types, and we have only covered some of the possibilities. Most methods come at the cost of removing at least some information from the original data. It remains to be seen however, if this cost is worth while to ensure adequate model perfomance and feasable training durations.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Possible Model Inputs*],  [*Dimensionality of Output*], [*Output Domain*],
    [Raw Onsource + Injection], [1], [Time],
    [Whitened Onsource + Injection], [1], [Time],
    [Pearsons Corrleation], [1], [Time],
    [Fourier Transform (PSD)], [1], [Frequency],
    [Power Spectral Density (PSD)], [1], [Frequnecy],
    [Spectrogram ], [2], [Time and Frequency]
  ),
  caption: [A non-exhaustive table of possible data conditioning modes. Feature engineering is often used in order to simplify a problem before it is presented to a machine learning model. There are many ways we could do this with gravitational-wave data. Presented are some of the most common. Each is described in more detail in @feature-eng-sec.]
) <feature-enginering-types>


=== Transient Glitch Simulation

As has previously been noted, as well as a quasi-stationary coloured Gaussian background, interferometer noise also contains transient detector glitches caused by a plethora of sources, both known and unknown. These glitches have a prominent effect on the upper-sensitivity bound of most types of search, so it may be important to represent features of this type in our training pipeline. Previous experiments perfomed during the development of the MLy pipeline @MLy have shown that networks can often have greatly increased FARs when performing inference on data segments which contain transient glitches, even when those glitches were only present in the off-source segment used to generate the PSD use for data whitening. As such, a method to add glitches to the training distribution should be considered so that methods to deal with features of this type can hopefully be incorporated into the model's learned parameters during training.

There have been multiple attempts to classify and document the many transient glitches found in real interferometer data @noise_clasification @dict_glitch_classifer @gravity_spy, both through automated and manual means @online_glitch_classification_review. During operation within a standard observing run, there are both intensive manual procedures @O2_O3_DQ to characterise the detector state and automated pipelines such as the iDQ pipeline @idq. There is also a large amount of work done offline to characterise the noise in a non-live environment @O2_O3_DQ. These methods utilise correlation with auxiliary channels, frequency of triggers, and other information about the detector state to ascertain the likelihood that a given feature is a glitch or of astrophysical origin.

One of the most prominent attempts to classify transient glitches is the Gravity Spy project @gravity_spy, which combines machine learning and citizen science to try and classify the many morphologies of transient glitches into distinct classes. Successful methods to classify glitches are highly useful since if a similar morphology appears again in the data it can be discounted as a probable glitch. Gravity Spy differentiates glitches into 19 classes plus one extra "no_glitch" class for noise segments that are proposed that do not contain a glitch. The other 19 classes are as follows: air_compressor, blip, chirp, extremely_loud, helix, koi_fish, light_modulation, low_frequency_burst, low_frequency_lines, none_of_the_above, paired_doves, power_line, repeating_blips, scattered_light, scratchy, tomte, violin_mode, wandering_line, and whistle. Some types, such as blips, are much more common than others.

There are two options we could use as example data in our dataset in order to familiarise the model with glitch cases. We could either use real glitches extracted from the interferometer data using the timestamps provided by the Gravity Spy catalogue @gravity_spy or simulated glitches we generate ourselves. The forms of each would vary depending on whether it was a multi, or single-detector example and whether we are attempting to detect CBCs or bursts.

*Real Glitches:* The addition of real glitches to the training dataset is a fairly intuitive process, though there are still some parameters which have to be decided upon. By using timestamps from the Gravity Spy catalogue @gravity_spy, we can extract time segments of equal length to our example segments, which contain instances of different classes of glitches. We should process these identically to our regular examples with the same whitening procedure and off-source segments. Real glitches have the distinct advantage that any model will be able to use commonalities in their morphology to exclude future instances; this is also, however, their disadvantage. If you train a model on specific morphologies, then the introduction of new glitch types in future observing runs, which may well be possible given the constant upgrades and changes to detector technology, then it may be less capable of rejecting previously unseen glitch types @gravity_spy. However, it is still possible that these glitches will help the model to reject anything other than the true type of feature it has been trained to recognise by weaning it off simple excess power detection.

*Simulated Glitches:* The other option is to use simulated glitches. The form of these glitches depends highly on the nature of the search, primarily because you wish to avoid confusion between the morphology of the feature you want your method to identify and your simulated glitches. For example, in a CBC search, you could use WNBs as simulated glitches, as their morphologies are entirely distinct, and there is no possibility of confusion. However, if we are using coherent WNBs across multiple detectors to train a model to look for coherence, then we must be careful that our glitch cases do not look indistinguishable from true positive cases, as this would poison the training pool by essentially mislabeling some examples. We could, in this case, use incoherent WNBs as simulated glitches as, ideally, we want our coherent search to disregard incoherent coincidences. This is the aproach taken by the MLy pipeline @MLy, as a method to train the models to reject counterexamples of cohernet features.

Other than the question of whether to use simulated or real glitches or maybe even both, a few questions remain: what is the ideal ratio between examples of glitches and non-glitched noise examples? Should the glitched background also be injected with waveforms at some rate? A real search would occasionally see glitches overlapping real signals, though this would occur in a relatively low number of cases, and including these types of signal-glitch overlaps could perhaps interfere with the training process whilst not adding a great deal of improvement to the true positive rate. Should glitches form their own class so that the model instead has to classify between signal, noise, or glitch rather than just signal or noise? These questions must be answered empirically.

For the multi-detector case, and thus also the burst detection case, we must decide how to align glitches across detectors. It seems safe to assume that adding coherent glitches across multiple detectors would be a bad idea in a purely coherence-based search pipeline --- although perhaps if the model can learn to disregard certain morphologies based on prior experience, this would be a nice extension. For some simple glitch types, coincident and fairly coherent instances across detectors are not extremely unlikely. For example in the case of the most common glitch class identified by GravitySpy @gravity_spy, blips, we often see coincident glitches in multiple detectors with a physically plausable arrival time difference, and because they are only glitches, their morphologies can often be similar.

We could also include cases of incoherent glitches across detectors but of the same class, incoherent glitches across detectors but of different classes, and any combination of glitches found in less than the full complement of detectors. Perhaps it would be the case that a good mix of all of these cases would better inoculate our model against glitches.

== Perceptron Validation Results

==== Training data

Now that we have finally assembled all the pieces required to generate training, testing, and validation datasets that can approximate real noise segments, we can finally repeat the experiments we performed on the MNIST data in @mnist-test-sec, with both single and multi-layer perceptrons. The model architectures are similar, though the input vectors are now be the size of our simulated interferometer output examples, `(NUM_EXAMPLES_PER_BATCH, NUM_SAMPLES)` in the case of the single detector CBC search and `(NUM_EXAMPLES_PER_BATCH, NUM_DETECTORS, NUM_SAMPLES)` in the multi-detector coherent burst search case. We will use values of `NUM_EXAMPLES_PER_BATCH = 32`, as this is a standard power of two value used commonly across artificial neural network literature, `NUM_DETECTORS = 2`, LIGO Hanford and LIGO Livingston, as this is the simplest possible network case, and because signals projected onto these two detectors will have a greater similarity than signals projected into either of these two detectors and Virgo detector, both due to sensitivity and orientation and position differences. We have set `NUM_SAMPLES = 2048` for the reasoning previously described in this chapter. A flattening layer will only be required in the multi-detector case, as for the single detector case, the input is already one-dimensional (The batch dimensions are not a dimension of the input data and simply allow for parallel processing and gradient descent; see @gradient-descent-sec). 

The obfuscating noise consisted of real data taken from LIGO Hanford and LIGO Livingston for each respective detector case. Locations of confirmed and candidate events are extracted from the data, but known glitch times have been left in training, testing, and validation datasets.

For the single CBC case cuPhenom waveforms with masses drawn from uniform distributions between #box("5.0" + h(1.5pt) + $M_dot.circle$) and #box("95.0" + h(1.5pt) + $M_dot.circle$) for the mass of both companions and between -0.5 and 0.5 for the dimensionless spin component are injected into the noise and scaled with SNR taken from a uniform distribution of between 8.0 and 15.0 unless explicitly stated.

For the multi-detector Burst case, a coherent WNB was injected with durations between #box("0.1"+ h(1.5pt) + "s") and #box("1.0" + h(1.5pt) + "s"), and the frequencies were limited to between #box("20.0" + h(1.5pt) + "Hz") and #box("500.0" + h(1.5pt) + "Hz"). The injected bursts were projected correctly onto the detectors using a physically realistic projection. The bursts were injected using the same scaling type and distribution as the CBC case, although notably, the network SNR was used rather than a single detector SNR.

During network training, the gradients were modified by batches consisting of 32 examples at a time, chosen as an industry standard batch size, and with a learning rate of 1.0 $times$ 10#super("-4"), and using the Adam optimiser, which again are common standards across the industry. During training epochs, 10^5 examples were used before the model was evaluated against the previously unseen validation data. It should be noted that unlike in standard model training practices,  due to the nature of the generators used for the training, no training examples were repeated across epochs, but the validation dataset was kept the same for each epoch. After each epoch, if the validation loss for that epoch was the lowest yet recorded, the model was saved, replacing the existing lowest model. If no improvement in validation loss was seen in 10 epochs (patience), the training was halted, and the best model was saved for further validation tests. @perceptron-training-parameters shows  a large number of the training and dataset hyperparameters.

#figure(
  table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon,
    [*Hyperparameter*],  [*Value*],
    [Batch Size], [32],
    [Learning Rate], [10#super("-4") ],
    [Optimiser], [ Adam ],
    [Scaling Method], [SNR],
    [Minimum SNR], [8.0],
    [Maximum SNR], [15.0],
    [SNR Distribution], ["Uniform"],
    [Data Acquisition Batch Duration], [2048.0 s],
    [On-source Duration], [1.0 s],
    [Off-source Duration], [16.0 s],
    [Scale Factor], [10#super("21") ],
    
  ),
  caption: [The common training and dataset hyperparameters shared by the CBC and Burst perceptron experiments. Note that the scale factor here refers to the factor used during upscaling of the CBC waveforms and noise, from their extremely small natural dimensions, to make them artifical neuron friendly. This is done both to ensure that the input values work well with the network activation functions and learning rates, which are tuned around values near one, and to reduce precision errors in areas of the code that use 32 bit precision, employed to reduce memory overhead, computational cost and duration.]
) <perceptron-training-parameters>

==== Architectures

We used four different layer architectures with zero, one, two, and three hidden layers; see @perceptron-cbc-architectures. All models have a custom-implemented whitening layer, which takes in two vectors, the onsource and offsource segments, and performs a whitening operation as described in @whitening-sec. They also all have a capping dense layer with a single output value that represents either the presence of a feature or the absence of one. The capping layer uses the SoftMax activation function; see @softmax, and the other hidden layers use ReLU activation functions, see @relu. 

Layers are built with a number of neurons selected from this list 64, 128, and 256, though fewer combinations were tested in architectures with more model layers. Models tested have these eight configurations of neural numbers per layer, specified as [num_hidden_layers:num_neurons_in_layer_1 ... num_layers_in_layer_n]: ([0], [1:64], [1:128], [1:256], [2:64,64], [2:128,64], [2:128,128], [3:64,64,64]).

#figure(
    image("perceptron_diagrams.png", width: 90%),
    caption: [Perceptron diagrams. The four different architectures used to test the use of purely dense models for both the single-detector CBC detection case and the multi-detector Burst detection problem. The only differences are that the input vector sizes were different between the cases: `(NUM_EXAMPLES_PER_BATCH, NUM_SAMPLES)` in the case of the single detector CBC search and `(NUM_EXAMPLES_PER_BATCH, NUM_DETECTORS, NUM_SAMPLES)` in the multi-detector coherent burst search case. All models take in two input vectors into a custom-designed whitening layer, the offsource and the onsource vector; see @whitening-sec for more information about the whitening procedure, and all models are capped with a dense layer with a single output neuron that is used to feed the binary loss function, with a sigmoid activation function. Each hidden layer has been tested with 64, 128, and 256 neurons:  _Top:_ Zero-hidden layer model. _Second to top:_ Two-hidden layer model. _Second to bottom:_ Three-hidden layer model. _Bottom:_ One hidden layer model. ]
) <perceptron-cbc-architectures>

=== CBC Detection Dense Results

First, we can examine the results of applying dense-layer perceptrons to the CBC single-detector morphology detection problem. 

=== Burst Detection Dense Results

Talk about the average SNR of the background at some point and how even Gaussian noise will generate a certain number of false alarms

Show that dense layers cant do it

introduce cnns and early work from the literature and show our recreations of their results as baseline values.

Talk about sample rate choice with justification at some point

== Introducing Convolutional Neural Networks (CNNs) <cnn-sec>
\
As we have seen, simple dense-layer perceptrons can not adequately perform detection tasks on gravitational-wave data. This was anticipated, given the complexity of the distribution. Perceptrons have not been at the forefront of artificial neural network science for some time. We must turn toward other architectures. Although, in some ways, specialising the network will limit the capacity of our model to act as a universal function approximator, in practice, this is not a concern, as we have at least some idea of the process that will be involved in completing the task at hand, in this case, image, or more correctly time-series recognition. 

The Convolutional Neural Network (CNN) is currently one of the most commonly used model archetypes. In many ways, the development of this architecture was what kickstarted the current era of artificial neural network development. On 30#super("th") December 2012, the AlexNet CNN @image_classification achieved performance in the ImageNet multi-class image recognition competition, far superior to any of its competitors. This success showed the world the enormous potential of artificial neural networks for achieving success in previously difficult domains.

CNNs are named for their similarity in operation to the mathematical convolution, although it is more closely analogous to a discrete cross-correlation wherein two series are compared to each other by taking the dot product at different displacements. Unless you are intuitively familiar with mathematical correlations, I do not think this is a useful point of reference for understanding CNNs. So, I will not continue to refer to convolutions in the mathematical sense past this paragraph.

CNNs are primarily employed for the task of image and time-series recognition. Their fundamental structure is similar to dense-layer networks on a small scale. They are comprised of artificial neurons that take in several inputs and output a singular output value after processing their inputs in conjunction with that neuron's learned parameters; see @artificial_neuron_sec. Typical CNNs ingest an input vector, have a single output layer that returns the network results, and contain a variable number of hidden layers. However, the structure and inter-neural connections inside and between the layers of a CNN are fundamentally different.

Unlike perceptrons, layers inside CNNs are, by definition, not all dense, fully-connected layers. CNNs introduce the concept of different types of sparsely-connected computational layers. The classical CNN comprises a variable number, $C$, of convolutional layers stacked upon the input vector, followed by a tail of $D$, dense layers, which output the result of the network. This gives a total of $N = C + D$ layers, omitting any infrastructure layers that may also be present, such as a flattening layer, which is often employed between the last convolutional layer and the first dense layer, because convolutional layers inherently have multidimensional outputs and dense layers do not. Purely convolutional networks, which consist only of convolutional layers, are possible, but these are a more unusual configuration, especially for classification tasks. Purely convolutional networks appear more often as autoencoders and in situations where you want to lessen the black-box effects of dense layers. Convolutional layers are often more interpretable than pure dense layers as they produce feature maps that retain the input vector's dimensionality.

Convolutional layers can and often do appear as layers in more complex model architectures, which are not necessarily always feed-forward models. They can appear in autoencoders, generative adversarial networks, recurrent neural networks, and as part of attention-based architectures such as transformers and generative diffusion models. We will, for now, consider only the classical design: several convolutional layers capped by several dense ones.

As discussed, CNNs have a more specialised architecture than dense layers. This architecture is designed to help the network perform in a specific domain of tasks by adding a priori information defining information flow inside the network. This can help reduce overfitting in some cases, as it means a smaller network with fewer parameters can achieve the same task as a more extensive dense network. Fewer parameters mean less total information can be stored in the network, so it is less likely that a model can memorise specific information about the noise present in training examples. A CNN encodes information about the dimensionality of the input image; the location of features within the input image is conserved as it moves through layers. It also utilises the fact that within some forms of data, the same feature is likely to appear at different locations within the input vector; therefore, parameters trained to recognise features can be reused across neurons. For example, if detecting images of cats, cat's ears are not always going to be in the same location within the image. However, the same pattern of parameters would be equally helpful for detecting ears wherever it is in the network.

The following subsections desribe different aspects of CNNs, including a description of pooling layers, which are companion layers often employed within convolutional networks.

=== Convolutional Layers

CNNs take inspiration from the biological visual cortex. In animal vision systems, each cortical neuron is not connected to every photoreceptor in the eye; instead, they are connected to a subset of receptors clustered near each other on the 2D surface of the retina. This connection area is known as the *receptive field*, a piece of terminology often borrowed when discussing CNNs.

*Convolutional Layers* behave in a similar manner. Instead of each neuron in every layer being connected to every neuron in the previous layer, they are only connected to a subset, and the parameters of each neuron are repeated across the image, significantly reducing the number of model parameters and allowing for translation equivariant feature detection. It is a common misnomer that convolutional layers are translation invariant; this is untrue, as features can and usually do move by values which are not whole pixel widths, meaning that even if the filters are the same, the pixel values can be different and give different results. One common problem with CNNs is that very small changes in input pixel values can lead to wildly different results, so this effect should be mitigated if possible. If they do not involve subsampling, however, CNNs are sometimes equivariant. This means that independent of starting location, ignoring edge effects, if you shift the feature by the same value, the output map will be the same --- this can be true for some configurations of CNN, but is also broken by most common architectures.

This input element subset is nominally clustered spacially, usually into squares of input pixels. This means that unlike with dense input layers, wherein 2D and greater images must first be flattened before being ingested, the dimensionality of the input is inherently present in the layer output. In a dense layer, each input is equally important to each neuron. There is no distinguishing between inputs far away from that neuron and inputs closer to that neuron (other than distinctions that the network may learn during the training process). This is not the case inside convolutional layers, as a neuron on a subsequent layer only sees inputs inside its receptive field. 

As the proximity of inputs to a neuron can be described in multiple dimensions equal to that of the input dimensionality, the network, therefore, has inherent dimensionality baked into its architecture --- which is one example of how the CNN is specialised for image recognition. In the case of a 2D image classification problem, we now treat the input vector as 2D, with the receptive field of each neuron occupying some shape, most simply a square or other rectangle, on the 2D vector's surface.

The term receptive field is usually reserved to describe how much of the input image can influence the output of a particular neuron in the network. The set of tunable parameters which define the computation of a neuron in a convolutional layer when fed with a subset of neuron outputs or input vector values from the previous layer is called a *kernel*. Each kernel looks at a subset of the previous layers' output and produces an output value dependent on the learned kernel parameters. A kernel with parameters tuned by model training is sometimes called a *filter*, as, in theory, it filters the input for a specific translation-invariant feature (although, as we have said, this is only partially true). The filter produces a strong output if it detects that feature and a weak output in its absence. Identical copies of this kernel will be tiled across the previous layer to create a new image with the same dimensionality as the input vector, i.e. kernels in a time-series classifier will each produce their own 1D time-series feature map, and kernels fed a 2D image will each produce a 2D image feature map. In this way, each kernel produces its own feature map where highly scoring pixels indicate the presence of whatever feature they have been trained to identify, and low-scoring ones indicate a lack thereof. Because the network only needs to learn parameters for this single kernel, which can be much smaller than the whole image and only the size of the feature it recognises, the number of trainable parameters required can be significantly reduced, decreasing training time, memory consumption, and overfitting risk. For a single kernel with no stride or dilation, see @stride-sec, applied to an input vector with no depth dimension, the number of trainable parameters is given by

$ op("len")(theta_"kernel") = (product_i^N S_i) + 1 $

where $op("len")(theta_"kernel")$ is the number of trainable parameters in the kernel, N is the number of dimensions in the input vector, and $S_i$ it the configurable hyperparameter, kernel size in the i#super("th") dimension. The extra plus one results from the bias of the convolutional kernel.
 
For example, a 1D kernel of size 3, would have $3 + 1 = 4$ total parameters, independent of the size of the input vector, and a 2D kernel of size $3 times 3$ would have $3 times 3 + 1 = 10$ total parameters, again independent of the size of the 2D input vector in either dimension.  See @kernel_example for an illustration of the structure of a convolutional kernel.

#figure(
    image("convolutional_kernel.png", width: 40%),
    caption: [Diagram of a single kernel, $""_1^1k$, in a single convolutional layer. In this example, a 1D vector is being input; therefore, the single kernel's output is also 1D. This kernel has a kernel size of three, meaning that each neuron receives three input values from the layer's input vector, $accent(x, arrow)$, which in this case is length five. This means there is room for three repeats of the kernel. Its parameters are identical for each iteration of $""_1^1k$ at a different position. This means that if a pattern of inputs recognised by the kernel at position 1, $""_1^1k_1$ is translated two elements down the input vector, it will be recognised similarly by the kernel at $""_1^1k_3$. Although this translational invariance is only strict if the translation is a whole pixel multiple and no subsampling (pooling, stride, or dilation) is used in your network, this pseudo-translational invariance can be useful, as often, in images and time series data, similar features can appear at different spatial or temporal locations within the data. For example, in a speech classification model, a word said at the start of the time series can be recognised just as easily by the same pattern of parameters if that word is said at the end of the time series (supposing it lies on the sample pixel multiple). Thus, the same kernel parameters and the same filter can be repeated across the time series, reducing the number of parameters needed to train the model. This particular kernel would have $3 + 1 = 4$ total parameters, as it applied to a 1D input vector, and has a kernel size of three, with an additional parameter for the neuron bias. With only a single kernel, only one feature can be learned, which would not be useful in all but the most simple cases. Thus, multiple kernels are often used, each of which can learn its own filter. ]
) <kernel_example>

When first reading about convolutional layers, it can be confusing to understand how they each "choose" which features to recognise. What should be understood is that this is not a manual process; there is no user input on which kernels filter which features; instead, this is all tuned by your chosen optimiser during the training process. Even the idea that each kernel will cleanly learn one feature type is an idealised simplification of what can happen during training. Gradient descent has no elegant ideas of how it should and should not use the architectures presented to it and will invariably follow the path of least resistance, which can sometimes result in strange and unorthodox uses of neural structures. The more complex and non-linear the recognition task, the more often this will occour. 

Although we do not specify exactly which features each kernel should learn, there are several hyperparameters that we must fix for each convolutional layer before the start of training. We must set a kernel (or filter) size for each dimension of the input vector. For a 1D input vector, we will set one kernel size per kernel; for a 2D input vector, we must set two, and so on. These kernel dimensions dictate the number of input values read by each kernel in the layer and are nominally consistent across all kernels in that layer; see @kernel-size for an illustration of how different kernel sizes tile across a 2D input.

#figure(
    image("kernel_sizes.png", width: 50%),
    caption: [Illustration of how different values of kernel size would be laid out on a $4 times 4$ input image. In each case, unused input image values are shown as empty black squares on the grid, and input values read by the kernel are filled red. The grids show the input combinations that a single kernel would ingest if it has a given size, assuming a stride value of one and zero dilation. The kernel sizes are as follows: _Upper left:_ $2 times 2$. _Upper right:_ $3 times 2$. _Lower left:_  $2 times 3$. _Lower right:_  $3 times 3$. One pixel in the output map is produced for each kernel position. As can be seen, the size of the output map produced by the kernel depends both on the input size and the kernel size; smaller kernels produce a larger output vector.]
) <kernel-size>

The other hyperparameters that must be set are the number of different kernels and the choice of activation function used by the kernel's neurons. These hyperparameters can sometimes be manually tuned using information about the dataset, i.e. the average size of the features for kernel size and the number of features for the number of kernels, but these can also be optimised by hyperparameter optimisation methods, which might be preferable as is often difficult to gauge which values will work optimally for a particular problem.

Multiple kernels can exist up to an arbitrary amount inside a single convolutional layer. The intuition behind this multitude is simply that input data can contain multiple different types of features, which can each need a different filter to recognise; each kernel produces its own feature map as it is tiled across its input, and these feature maps are concatenated along an extra *depth* dimension on top of the dimensionality of the input vector. A 1D input vector will have 2D convolutional layer outputs, and a 2D input vector will result in 3D convolutional outputs. The original dimensions of the input vector remain intact, whilst the extra discrete depth dimension represents different features of the image; see @multi_kernel_example. 

In the case of a colour picture, this depth dimension could be the red, green, and blue channels, meaning this dimension is already present in the input vector. The number of trainable parameters of a single convolutional layer is given by

$ op("len")(theta_"conv_layer") = K times ((D times product_i^N S_i) + 1) $ <conv-layer-size>

where $op("len")(theta_"conv_layer")$ is the total number of parameters in a convolutional layer, $K$ is the number of convolutional kernels in that layer, a tunable hyperparameter, and $D$ is the additional feature depth dimension of the layer input vector, which is determined either by the number pre-existing feature channels in the input vector, i.e. the colour channels in a full-colour image or, if the layer input is a previous convolutional layer, the number of feature maps output by that previous layer, which is equivalent to the number of kernels in the previous layer. For example, a 1D convolutional layer with three kernels, each with size three, ingesting a 1D input with only a singleton depth dimension would have $3 times ((1 times (3)) + 1) = 12$ total trainable parameters, whereas a 2D convolutional layer with three kernels of size $3 times 3$ looking at a colour RGB input image would have $3 times (3 times ( 3 times 3 ) + 1) = 84$ total trainable parameters.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("multiple_convolutional_kernel.png",   width: 45%) ],
        [ #image("single_conv_abstraction.png", width: 100%) ],
    ),
    caption: [_Upper:_ Diagram of three convolutional kernels, $[""_1^1k, ""_2^1k, ""_3^1k]$, in a single convolutional layer. Each kernel is coloured differently, in red, green, and blue. Artificial neurons of the same colour will share the same learned parameters. Again, a 1D vector is being input; therefore, the output of each of the kernels is 1D, and the output of the kernels stack to form a 2D output vector, with one spatial dimension retained from the input vector and an extra discrete depth dimension representing the different features learned by each of the kernels. Again, each kernel has a kernel size of three. Multiple kernels allow the layer to learn multiple features, each of which can be translated across the input vector, as with the single kernel. Using @conv-layer-size, this layer would have $3 times ((1 times 3) + 1) = 12$ trainable parameters. It should be noted that this is a very small example simplified for visual clarity; real convolutional networks can have inputs many hundreds or thousands of elements long and thus will have many more iterations of each kernel, as well as many more kernels sometimes of a much larger size. _Lower:_ Abstracted diagram of the same layer with included hyperparameter information. ]
) <multi_kernel_example>

As with dense layers, multiple convolutional layers can be stacked to increase the possible range of computation available; see @multi_cnn_layer_example. The first convolutional layer in a network will ingest the input vector, but subsequent layers can ingest the output of previous convolutional layers, with kernels slicing through and ingesting the entirety of the depth dimension. In theory, this stacking allows the convolutional layers to combine multiple more straightforward features in order to recognise more complex, higher-level features of the input data --- although, as usual, things are not always quite so straightforward in practice. When calculating the number of trainable parameters in multiple convolutional layers, we can simply use @conv-layer-size for each layer and sum the result.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("two_convolutional_layers.png", width: 60%) ],
        [ #image("multi_conv_abstraction.png", width: 100%) ],
    ),
    caption: [_Upper:_ Diagram of two convolutional layers, each with independent kernels. The first layer has three kernels, each with a size of three. The second layer has two kernels, both with a size of two. Again, this is a much-simplified example which would probably not have much practical use. Different kernels are coloured differently, in red, green, and blue. Although it should be noted that similar colours across layers should not be taken as any relationship between kernels in different layers, they are each tuned independently and subject to the whims of the gradient descent process. This example shows how the kernels in the second layer take inputs across the entire depth of the first layer but behave similarly along the original dimension of the input vector. In theory, the deeper layer can learn to recognise composite features made from combinations of features previously recognised by the layers below and visible in the output feature maps of the different kernels. This multi-layer network slice would have $(3 times ((1 times 3) + 1)) + (2 times ((3 times 2) + 1)) = 26$ total trainable parameters. This was calculated by applying @conv-layer-size to each layer. _Lower:_ Abstracted diagram of the same layers with included hyperparameter information. ]
) <multi_cnn_layer_example>

The result of using one or more convolutional layers on an input vector is an output vector with an extra discrete depth dimension, with each layer in the stack representing feature maps. Whilst often considerably more interpretable than maps of the parameters in dense layers, these maps are often not very useful alone. However, a flattened version of this vector is now, hopefully, much easier for dense layers to classify than the original image. As such, CNNs used for classification are almost always capped by one or more dense layers in order to produce the final classification result; see @cnn_diagram for a toy example of a CNN used for binary classification.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("cnn_diagram.png",   width: 100%) ],
        [ #image("cnn_abstracted.png", width: 100%) ],
    ),
    caption: [_Upper:_ Diagram of a very simple convolutional neural network binary classifier consisting of four layers with tunable parameters plus one infrastructure layer without parameters. Two subsequent convolutional layers ingest the five-element input vector, $accent(x, arrow)$; then the 2D output of the latter of the two layers is flattened into a 1D vector by a flattening layer. This flattened vector is then ingested by two dense layers, the latter of which outputs the final classification score. The first convolutional layer has three convolutional kernels, each with a size of three, and the second convolutional layer has two kernels, both with a size of two. The first dense layer has three artificial neurons, and the final output dense layer has a number of neurons dictated by the required size of the output vector. In the case of binary classification, this is either one or two. Different kernels within a layer are differentiated by colour, in this case red, green, or blue, but similar colour between layers does not indicate any kind of relationship. Dimensionless neurons are shown in black; it should be noted that after flattening, dimensional information is no longer necessarily maintained by the network structure. Of course, no information is necessarily lost either, as the neuron index itself contains information about where it originated, so during training, this information can still be used by the dense layers; it is just not necessarily maintained as it is in convolutional layers. This network will have in total $26 + (3 times 4 + 4) + (2 times 3 + 2) = 50$ trainable parameters. This network is very simple and would probably not have much practical use in real-world problems other than straightforward tasks that would probably not necessitate using neural networks. _Lower:_ Abstracted diagram of the same model with included hyperparameter information. 
 ]
) <cnn_diagram>

=== Stride, Dilation, and Padding <stride-sec>

* Stride * is a user-defined hyperparameter of convolutional layers which must be defined before training. By default, this will be set to one and often remains this way. Like kernel size, it is a multidimensional parameter with a value for each input vector dimension. A convolutional layer's stride describes the distance the kernel moves between instances. For example, if the stride is one, then a kernel is tiled with a separation of one input value from its last location. Stride, $S$, is always greater than zero, $S > 0$. The kernels will overlap in the i#super("th") dimension if $S_i < k_i$. If $S_i = k_i$, there will be no overlap and no missed input vector values. If $S_i > k_i$, some input vector values will be skipped; this is not usually used. Along with kernel size, stride determines the output size of the layer. A larger stride will result in fewer kernels and, thus, a smaller output size; see @stride below for an illustration of different kernels strides.

#figure(
    image("stride.png", width: 70%),
    caption: [Illustration of how different values of kernel stride would be laid out on a $4 times 4$ input image. In each case, unused input image values are shown as empty black squares on the grid, and input values read by the kernel are filled in red. Similar to kernel size, different values of stride result in a different output vector size. The strides shown are as follows: _Upper left:_ $2, 2$. _Upper right:_ $3, 2$. _Lower left:_  $2, 3$. _Lower right:_  $3,  3$.]
) <stride>

Introducing kernel stride primarily serves to reduce the overall size of your network by reducing the output vector without adding additional parameters; in fact, the number of parameters is independent of stride. Reducing the size of your network might be a desirable outcome as it can help reduce computational time and memory overhead. It can also help to increase the receptive field of neurons in subsequent layers as it condenses the distance between spatially separated points, so if you're playing with the resolution of feature maps in your model to balance the identification of smaller and larger scale features, it could potentially be a useful dial to tune. In most cases, however, it's left at its default value of one, with the job of reducing the network size falling to pooling layers; see @pool-sec. 

One interesting and potentially unwanted effect of introducing stride into our network is that it removes the complete translation equivariance of the layer by subsampling; instead, translations are only equivariant if they match the stride size, i.e. if a kernel has a stride of two features are invariant if they move exactly two pixels, which is not a common occurrence.

*Dilation* is a further hyperparameter that can be adjusted prior to network training; by default, this value would be set to zero, and no dilation would be present. Dilation introduces a spacing inside the kernel so that each input value examined by the kernel is no longer directly adjacent to another kernel input value, but instead, there is a gap wherein that element is ignored by the kernel. This directly increases the receptive field of that kernel without introducing additional parameters, which can be used to help the filters take more global features into account. It can also be used in the network to try and combat scale differences in features; if multiple kernels with different dilations are used in parallel on different model branches, the model can learn to recognise features at the same scale but with different dilations; see @dilation.

#figure(
    image("dilation.png", width: 40%),
    caption: [Diagram illustrating how different values of kernel dilation affect the arrangement of the kernel input pixels. In this example, the receptive field of a single $3 times 3$ kernel at three different dilation levels is displayed; differing colours represent the input elements at each dilation level. The shaded red kernel illustrates dilation level zero; the shaded blue region is a kernel with a dilation of one, and the green kernel has a kernel dilation of two.]
) <dilation>

Particular stride, dilation, and size combinations will sometimes produce kernel positions that push them off the edge of the boundaries of the input vector. These kernel positions can be ignored, or the input vector can be padded with zeros or repeats of the nearest input value; see @padding.

#figure(
    image("padding.png", width: 40%),
    caption: [Diagram illustrating how padding can be added to the edge of an input vector in order to allow for otherwise impossible combinations of kernel, stride, size, and dilation. In each case, unused input image values are shown as empty black squares on the grid, input values read by the kernel are shaded red, and empty blue squares are unused values added to the original input vector, containing either zeros or repeats of the closest data values. In this example, the kernel size is $3 times 3$, and the kernel stride is $2, 2$.]
) <padding>

=== Pooling <pool-sec>

Pooling layers, or simply pooling, is a method used to restrict the number of data channels flowing through the network. They see widespread application across the literature and have multiple valuable properties. They can reduce the size of your network and thus the computation and memory overhead, and they can also make your network more robust to small translational, scale, and rotational differences in your network. Convolutional layers record the position of the feature they recognise but can sometimes be overly sensitive to tiny shifts in the values of input pixels. Small changes in a feature's scale, rotation or position within the input can lead to a very different output, which is evidently not often desirable behaviour.

Pooling layers do not have any trainable parameters, and their operation is dictated entirely by the user-selected hyperparameters chosen before the commencement of model training. Instead, they act to group together pixels via subsampling throwing away excess information by combining their values into a single output. In this way, they are similar to convolutional kernels, however. instead of operating with trained parameters, they use simple operations. The two most common types of pooling layers are *max pooling* and *average pooling*; max pooling keeps only the maximum value within each of its input bins, discarding the other values; intuitively, we can think of this as finding the strongest evidence for the presence of the feature within the pooling bin and discarding the rest. Average pooling averages the value across the elements inside each pooling bin, which has the advantage that it uses some information from all the elements.

As can be imagined, the size of CNNs can increase rapidly as more layers and large numbers of kernels are used, with each kernel producing a feature map nearly as large as its input vector. Although the number of parameters is minimised, the number of operations increases with increasing input size. Pooling layers are helpful to reduce redundant information and drastically reduce network size whilst also making the network more robust to small changes in the input values.

Along with the choice of operational mode, i.e. average or maximum, pooling layers have some of the same hyperparameters as convolutional kernels, size and stride. Unlike convolutional layers, the pooling stride is usually set to the same value as the pooling size. Meaning that there will be no overlap between pooling bins but also no gaps. This is due to the purpose of pooling layers, which attempt to reduce redundant information; if stride were set to smaller values, there would be little reduction and little point to the layer.

Because pooling with stride is a form of subsampling, it does not maintain strict translational equivariance unless the pool stride is one, which, as stated, is uncommon. Thus, as most CNN models use pooling, most CNNs are neither strictly translationally invariant nor equivariant.

== Results from the Literature

CNNs are introduced here as a thorough literature review of other attempts to solve the classification problem would be prudent before we continue. Almost all attempts from the literature involve models which are at least as complex as CNNs, so an understanding of their operation is crucial to gaining an effective overview of the current state of the field. This is not intended to be an exhaustive catalogue, although efforts have been made to be as complete as possible.

Both gravitational-wave astrophysics and deep learning methods have been through rapid advancement in the previous decade, so it is perhaps unsurprising that there has also developed a significant intersection between the two fields. Multiple artificial network architectures, CNNs, autoencoders, generative adversarial networks, recurrent neural networks, and attention-based networks like transformers and generative diffusion models have been applied to numerous gravitational-wave data analysis problems. This review will focus on efforts made to apply CNN classifiers to detect features hidden within interferometer data. First we will look at attempts to detect Compact Binary Coalescences (CBCs) followed by a look unmodeled Bursts detection attempts. More complex network architectures will be reviewed later, when we examine attention layers in closer detail; see @skywarp-sec.

The earliest attempts at CBC classification using artificial neural networks were a pair of independently published papers by George et al. @george_huerta_cnn and Gabbard et al. @gabbard_messenger_cnn. George et al. @george_huerta_cnn applied CNNs to both the binary classification problem and basic parameter estimation. They used CNNs with two outputs to extract parameter estimates for the two companion masses of the binary system. They used the whitened outputs of single interferometers as inputs and utilised CNNs of a standard form consisting of convolutional, dense, and pooling layers. They evaluated two models, one smaller and one larger. In their first paper, they used only simulated noise, but they produced a follow-up paper showing the result of the model's application to real interferometer noise @george_huerta_followup. George et al. used a alternate CNN design with a different combination of layers. They only used a single network architecture, and no attempt at parameter estimation was made. A differentiating feature of their paper was the training of individual network instances to recognise different SNRs. Both George et al. @george_huerta_cnn and Gabbard et al. @gabbard_messenger_cnn achieved efficiency curves that closely resembled that of matched filtering; of note, however, is that both were validated at a considerably higher FAR than is usually allowed in a production search, this will be a consistent theme throughout the literature and is one of the greatest blockers to using CNNs in an official search pipeline.

There have been many papers that follow up on these two initial attempts. Several papers with mixed results are hampered by inconsistencies and unclear methodology. Luo et al. @luo_cnn attempted to improve the model described by Gabbard et al. They have presented their results using a non-standard "Gaussian noise amplitude parameter" and in the form of Receiver Operator Curves (ROC) on a linear rather than a log scale. Whilst within their own comparisons, they seem to have improved network operation over the original design, at least at higher FARs, it is difficult to make a comparison against other papers because of the unorthodox presentation. Schmitt et al. @schmitt_cnn attempted to compare the performance of one of the models presented in George et al. @george_huerta_cnn with three different model architectures, Temporal Convolutional Networks (TCNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTMs). However, the paper has a strange use of SNR, which is not described in detail; they used negative SNRs, which would be non-physical if it were describing optimal SNRs alogn with SNRs of zero, which should only create entirely zero-valued waveforms. They seem to show that the other model architectures can achieve higher performance than CNNs, but without a known waveform scaling, it is hard to compare to other results. 

A more interesting follow-up by Fan et al. @multi_detector_fan_cnn took the smaller of the two models introduced in George et al. @george_huerta_cnn and extended it to use multiple detectors as inputs rather than the previously mentioned studies, which looked at only single detectors. They do this for both detection and parameter estimation and appear to show improved accuracy results over the original paper @george_huerta_cnn, although they do not address the confounding factor of having to deal with real noise. Krastev et al. tested the use of the other larger model introduced by George et al. @george_huerta_cnn. They tested its use on Binary Neuron Star (BNS) signals, as well as reaffirming its ability to detect BBH signals. They used significantly longer input windows to acount for the longer detectable duration of BNS signals. They found BNS detection to be possible, although it proved a significantly harder problem. 

Using a different style of architecture, Gebhard et al. @gebhard_conv_only_cnn argued that convolution-only structures are more robust and less prone to error, as they remove much of the black-box effect produced by dense layers and allow for multiple independently operating (though with overlapping input regions) networks, creating an ensemble which generates a predictive score for the presence of a signal at multiple time positions. This results in a time-series output rather than a single value, which allows the model to be agnostic to signal length. Their determination of the presence of a signal can thus rely on the overall output time series rather than just a single classification score. Similarly to Fan et al. @multi_detector_fan_cnn, they used multiple detector inputs. Whilst this is interesting work, they give their results at different values of False Positive Rate (FPR), the probability that a given positive result is erroneous, rather than False Alarm Rate (FAR), the probability that a given duration of noise with no signal will produce a positive result, so again it is hard to make a direct comparison to other studies. 

There have been at least two papers which utalise ensemble aproaches to the problem. Ensembles of model consist of multiple inpendently trained models, in the hopes that the weaknesses of one will be counteracted by the strengths of another under the assumption that it is less likely for them both to be weak in the same area. A joint decision is then taken through some mecahnism that takes the result of all models into consideration, often waiting certain models votes under certain criteria. Huerta et al. @huerta_fusion_cnn used an aproach consisting of four independently trained models, each of which has two sepratated CNN branches for the LIGO Hanford and LIGO Livingston detectors, which are then merged by two further CNN layers. Their efficiency results are presented on a logarithmic ROC, which gives more clarity that a linear one, but their results are stil clusted tightly in one corner makeing them hard to parse. Still, they have efficiency results down to a lower FAR than any paper reviewed so far, at $1 times 10^(-5)$, which is impressive, although the efficiency scores at these FARs are low ($<1%$). Overall, the paper is more focused on the software infrastructure for deploying neural network models. Ma et al. @ma_ensemble_cnn used an ensemble networks that employ one of the architectures described by Gabbard et al. @gabbard_messenger_cnn. They utilise two "subensembles" in an arrangement in which each detector has its own ensemble composed of networks which vote on a false/positive determination; the results of both of the two subensembles are then combined for a final output score. They do not give efficiency scores at set SNRs, so again, it is difficult to compare against other results.

There have also been some interesting studies which use feature engineering to extract features from the input data before those features are fed into the CNN models, see @feature-eng-sec. Wang et al. @wang_cnn use a sparse matched filter search, where template banks of only tens of features, rather than the usual hundreds of thousands or millions, were used. The output of this sparce matched filter was then ingested by a small CNN, which attempted to classify the inputs. Notably, they use real noise from the 1#super("st") LVK joint observing run and multi-detector inputs. Though an interesting method, their results appear uncompetitive with other approaches. Reza @matched_filtering_combination et al. used a similar approach but split the input into patches before applying the matched filter. However, results are not presented in an easily comparable fashion. Bresten et al. @bresten_cnn_topology adapts one of the architectures from George et al. @george_huerta_cnn but applies a feature extraction step that uses a topological method known as persistent homology before the data is ingested by the network. It is an interesting approach, but their results are unconvincing. They limited their validation data to 1500 waveforms at only 100 specific SNR values in what they term their "hardest case". They showed poor results compared to other methods, suggesting their method is undeveloped and heavily SNR-tailored. 

There have been at least three spectrogram-based attempts to solve the CBC detection problem: Yu et al. @spectrogram_cnn_2 and Aveiro et al. @bns_object_detection_spectogram. Yu et al. used single detector spectrograms, which are first analysed in strips using multiple 1D CNNs before being fed into a 2D CNN for final classification; they achieve middle-of-the-range efficiency results. Aveiro et al. @bns_object_detection_spectogram focus on BNS detection and used an out-of-the-box object detection network to try and detect patterns in spectrograms. They do not state efficiencies for SNRs less than ten. Finally, there was also a search paper @o2_search_cnn, which searched through the second observing run using spectrograms-based CNNs; they detected nothing of significance.

There has also been an attempt to use wavelet decomposition for the problem. Lin et al. @lin_wavelet_bns focused on the detection of BNS signals by wavelet decomposition with some very promising results shown to outperform matched filtering; a subsequent follow-up paper @li_wavelet_cnn showed that the same method could be applied to BBH signals with equal promise. They achieve an efficiency of 94% when detecting waveforms with an SNR of 2 at a FAR of $1 times 10^(-3)$, which undercuts the competition by considerable margins. Their method is certainly worth investigation but was unfortunately missed until this thesis was in the latter stages of construction, so no wavelet decomposition methods have been attempted.

There have also been a number of papers utilising CNNs for specialised detection cases, such as mass asymmetric CBCs @mass_asymetric_cbcs by Andrés-Carcasona et al., who employ spectrogram-based CNNs to run a search over O3, and eccentric CBCs by Wei et al. @wei_cnn, the latter of which also focuses on early detection along with a few other papers @early_alert_bbh @early_alert_bns @early_detection which attempt to detect CBCs signals before the inspiral proper. There have also been a number of papers which discuss the use of CNNs for the analysis of data from future space-based detectors @space_detection @space_detection_2. For brevity, and as they are less relevant to our problems, these special cases will not be discussed here. 

As can be seen, it is very difficult to compare the performance of many of the architectures and methods presented in the literature. The results are presented at wildly different FARs and SNR ranges, often using different incomparable metrics and with varying levels of rigour. There is a tendency to apply new tools and ideas to the problem without careful thought about how the results can be standardised. @literature-results displays results from some of the papers which were found to have at least somewhat comparable metrics.

#set page(
  flipped: true
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Name*],  [*Conv*], [*Pool *], [*Dense*], [*Real Noise?*], [*Detectors*], [*Target*], [*Feature*], [*SNR Tailored*], [*FAR*], [*Acc 8*], [*Acc 6*], [*Acc 4*],
    [George et al. @george_huerta_cnn], [0, 2, 4], [1, 3, 5], [6, 7], [No], [Single], [BBH], [No], [No], [$5 times 10^(-2)$], [0.98], [0.70], [0.16],
    [-], [0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10], [-], [-], [-], [-], [-], [-], [0.99], [0.80], [0.21],
    [George et al. @george_huerta_followup], [-], [-], [-], [ #text(red)[*Yes*] ], [-], [-], [-], [-], [-], [0.98], [0.77], [0.18],
    [Gabbard et al. @gabbard_messenger_cnn], [0, 1, 3, 4, 6, 7], [2, 5, 8], [9, 10, 11], [No], [Single], [BBH], [No], [#text(red)[*Yes*]], [$1 times 10^(-1)$], [1.0], [0.88], [0.44],
    [-], [-], [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-2)$], [0.99], [0.69], [0.10],
    [-], [-], [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-3)$], [0.98], [0.49], [0.02],
    [Fan et al. @multi_detector_fan_cnn], [0, 2, 4], [1, 3, 5], [6, 7], [No], [#text(red)[*Three*]], [BBH], [No], [No], [$4 times 10^(-2)$], [0.99], [0.84], [0.32],
    [Krastev et al. @krastev_bnn_cnn ], [0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10], [No], [Single], [#text(red)[*BNS*]], [No], [No], [$1 times 10^(-1)$], [0.71], [0.42], [0.20],
    [- ], [-], [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-2)$], [0.32], [0.10], [0.02],
    [- ], [-], [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-3)$], [0.11], [0.00], [0.00],
    [Gebhard et al. @gebhard_conv_only_cnn], [#text(red)[*Conv layers $times 12$*]], [None], [None], [#text(red)[*Yes*]], [#text(red)[*Two*]], [BBH], [No], [No], [800 (IFPR)], [0.83], [0.35], [Not Given],
    [-], [-], [-], [-], [-], [-], [-], [-], [-],[450 (IFPR)], [0.80], [0.32], [Not Given],
    [-], [-], [-], [-], [-], [-], [-], [-], [-], [100 (IFPR)], [0.67], [0.23], [Not Given],
    [Wang et al. @wang_cnn], [0, 2], [1, 3], [4, 5], [#text(red)[*Yes*]], [#text(red)[*Two*]], [BBH], [#text(red)[*Matched Filter*]], [No], [$1 times 10^(-1)$], [0.60], [0.24], [0.12],
    [-], [-], [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-2)$], [0.30], [0.05], [0.00],
    [-], [-], [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-3)$], [0.08], [0.00], [0.00],
    [Huerta et al. @huerta_fusion_cnn], [#text(red)[*Ensemble $times 4$*]], [None], [-], [Yes], [Two], [BBH], [No], [No], [#text(red)[*$5 times 10^(-4)$*]],  [0.20], [0.15], [Not Given],
    [-], [-], [-], [-], [-], [-], [-], [-], [-], [#text(red)[*$5 times 10^(-5)$*]], [0.01], [0.001], [Not Given],
    [Yu et al. @spectrogram_cnn_2], [$2 times 1D "CNN"$  $->$  $2D "CNN"$], [-], [-], [Yes], [Single], [BBH], [#text(red)[*Spectrogram*]], [No], [$6 times 10^(-2)$], [0.89], [0.67], [0.20] 
  ),
  caption: [Note: Some accuracy values are extracted from plots by eye, so substantive error will have been introduced.]
) <literature-results>

#set page(
  flipped: false
)


The literature on burst detection with CNNs is, fortunately, much more limited. In all of the previously mentioned deep-learning studies, the training of the network has relied on accurate models of CBC waveforms. As has been noted, the availability of simulated waveforms for other potential gravitational-wave sources, i.e. bursts, is considerably narrower, both due to unknown physical processes and large numbers of free parameters, making it impossible to have a sufficiently sampled template bank.

Despite this, there have been some attempts, most notably using simulated supernovae waveforms, which are the most likely burst sources to be detected first. There have been at least five attempts to classify supernovae with this method @supernovae_cnn_1 @supernovae_cnn_2 @supernovae_cnn_3 @supernovae_cnn_4 including one utilising spectrograms for feature engineering @supernovae_spectrogram.  



In each of these deep-learning studies, the training of the network has relied on the availability of highly accurate models for  
gravitational wave signals from binary mergers cite{Blanchet_LRR}.
The application of deep learning methods to more general types of gravitational-wave signals has been more limited, with  core-collapse supernovae being the most prominent example.
These studies have made use of either catalogs of gravitational-wave signals from numerical simulations of core collapse or phenomenological waveform models fit to such catalogs.
%
For example, Chan textit{et al.}~cite{PhysRevD.102.043022} trained a CNN using simulated gravitational-wave timeseries with core collapse supernovae signals drawn from a range of published catalogs covering both magnetorotational-driven and neutrino-driven supernovae, and measured the ability to both detect the signal and correctly classify the type.
%
Iess textit{et al.}~cite{iess2020corecollapse} considered the problem of distinguishing true signals from noise fluctuations (``glitches'') that are common in real detectors. They used signals drawn from numerical catalogs combined combined with a simple phenomenological models for two glitch types to train CNNs to distinguish supernova signals from noise glitches.
%
Lopez~textit{et al.}~cite{PhysRevD.103.063011} (building on cite{PhysRevD.98.122002})
used a phenomenological model mimicking gravitational-wave signals from non-rotating core-collapse supernovae to train 
a complex mini-inception resnet neural network cite{7780459} to detect supernova signals in time-frequency images of LIGO-Virgo data.

We note that all of these examples, both for binary mergers and for supernovae, rely on having a signal model to train the deep network. 
As a consequence, their applicability is restricted to signals that are similar to the training data.  While not an issue for binary mergers, this may be very important for supernovae where the simulations used for training rely on uncertain physics and numerical approximations and simplifications  cite{Ott_2009,Abdikamalov2020}. And they are clearly not applicable to the more general problem of detecting gravitational-wave transients from as-yet unknown sources.

%Other studies primarily restricted to CCSNe and use numerical simulations or phenomenological fits to those simulations for training.
%There has been somewhat less interest in the application of deep learning to the detection of other, as yet undetected, gravitational-wave sources, although there have been a number of studies into the efficacy of using ANNs to detect supernovae waveforms cite{PhysRevD.98.122002,iess2020corecollapse}. 
%This has been made possible primarily through a number of recent attempts to produce supernovae gravitational-wave templates through multidimensional simulation cite{Powell:2018isq, radice2019characterizing, andresen2019gravitational, takiwaki2018anisotropic}. 
%However it should be noted that some of these utilise approximations as the required physics is much less understood than in the CBC case.

%There are two natural ways to input the data into CNN models, as 1D time series or as 2D time-frequency maps. The most obvious is to input the 1D time series - this preserves the full data output of the detectors. A 2019 paper by Chan et al. \cite{PhysRevD.102.043022} applied convoloutional neural networks to simulated gravitational wave time series with core collapse supernovae injections. They used a deep convoloutional network consisting of 8 convoloutional layers and 3 dense layers, and input all four gravitational wave detectors of the LIGO-VIRGO-Kagra network as well as all four with the two LIGO detectors in Advanced LIGO configurations. Their False Alarm Probability was set at 10\% a much higher false alarm rate than is being presented here, and focused only on supernovae waveforms. They tested four different supernovae waveforms, notably not included in the training bank, and achieved accuracies between 53\% and 90\%.

%The second method, inputting the data as 2D time-frequency maps, is closer to the most successful domain of CNNs: image recognition. There have been several papers which investigate this method's application to supernovae. A 2020 paper by Iess et al. \cite{iess2020corecollapse} investigates the application of CNNs to both 1D gravitational wave time-series and 2D time-frequency plots of simulated Virgo and Einstein telescope noise with injected simulations of neutrino-driven core-collapse supernovae. They also attempt to account for transient glitches by including training examples with injected sine-Gaussian's and a approximated representations of scattered light glitches. They achieve a single detector accuracy of over 95\% for both 1D and 2D CNN pipelines. Instead of running the CNN across all data they use a Wavelet detection filter to generate triggers and a value for the SNR.

%2D time-frequency maps can also be inputted in multi-detector configurations, similar to the 1D case. Each detector can be added to the depth channel in a similar way to how the RGB channels are usually handled in generic image classification. An example of this can be seen in a 2018 paper by Astone et al. cite{ PhysRevD.98.122002} wherein a CNN is used to classify multi-channel time-frequency images of the two LIGO detectors and the Virgo detector. Rather than using a hydro-dynamically simulated template bank, they opt for a engineered waveform which can replicate many of the features seen in simulated waveforms - this allows for a larger exploration of possible parameter space. They measured results at a FAR of $7 times 10^{-5}$ Hz With efficiencies above 90\% at an SNR of 20. 
%A 2020 paper by Lopez et al. cite{PhysRevD.103.063011} introduces a complex mini-inception resnet neural network to detect waveforms found in time-frequency images of 3D simulations of core collapse supernovae. Again three detectors were used with each detector layered as depth in the time frequency image. They achieved a detection efficiency of 70\% at a False Alarm rate of 5\%, on signals injected into real noise data from O2. 

Shortly after the release of an early version of this work~cite{Skliris:2020qax}, Marianer \textit{et al.}~\cite{semisuper} presented a deep-learning algorithm that avoids relying on a signal model by instead using outlier detection. The authors trained a mini-inception resnet network \cite{7780459} on the Gravity Spy data set cite{Zevin_2017,gravity_spy_2018}, which contains spectrograms of known noise glitches classified into categories. They then applied the CNN to spectrograms of LIGO data and used two methods of outlier detection to identify possible signals. This search was applied to a subset of public LIGO data from the first two observing runs; no signal candidates were found.
To our knowledge this is the only other case to date of a deep-learning method capable of searching for generic gravitational-wave transients.

In this paper we present a deep-learning technique that is capable of detecting generic transient gravitational-wave signals. Our approach differs from previous approaches in a key way: rather than training a CNN to recognise specific signal morphologies in the data streams, we construct CNNs that are designed to recognise textit{coherence} in amplitude and phase  between two or more data streams. We then train the CNNs using simulated signals and noise glitches that both consist of \textit{random} timeseries with properties drawn from the same distributions. 
Using the same waveform distributions to simulate both the  signals and glitches prevents the CNNs from using the signal morphology to the classify input. 
Instead, the CNNs are forced to learn to measure consistency between detectors.

In the next section we describe the architecture of textsc{MLy}'s CNNs and the training procedure. We then evaluate \textsc{MLy} by analysing data from the second LIGO-Virgo observing run. We will see that 
our trained pipeline has a detection efficiency approaching that of the standard LIGO-Virgo pipeline for detecting unmodelled gravitational-wave transients \cite{Klimenko:2015ypf}, but with higher speed and lower computational cost.
