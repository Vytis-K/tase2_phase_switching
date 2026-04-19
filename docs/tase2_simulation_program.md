# Simulation Program for Current-Pulse-Induced State Evolution in TaSe\(_2\)

## Purpose and scope

The purpose of the simulation work is not to produce a decorative numerical companion to the experiment, but to construct a sequence of models that separate competing physical explanations for the spatial patterns observed after current pulsing. The experimental problem contains several coupled effects that cannot be disentangled by visual inspection alone. The measured nano-ARPES maps encode genuine changes in electronic structure, geometrical variations in the local orientation of the sample surface, mixed spectra from boundary regions, and the consequences of nonuniform current flow and Joule heating. A useful simulation program must therefore be designed to answer specific questions rather than to reproduce every aspect of the experiment from the outset.

The central problem is that the observed written and erased patterns could arise from several different mechanisms. Some spatial variation may be caused by the electronic phase of the material itself. Some may be inherited from the morphology of the cleaved sample, including steps, terraces, flakes, and local tilts that change the measured momentum-space cut. Some may result from current crowding or nonuniform heat dissipation. Some may reflect interface-controlled kinetics in which the fate of a region depends strongly on its distance from a boundary between a metastable metallic state and the equilibrium charge-density-wave state. The role of simulation is to determine which of these mechanisms are capable of generating the experimental signatures and which are not.

A successful simulation program should therefore proceed from the simplest models capable of isolating a single mechanism to more coupled models that combine geometry, transport, thermal diffusion, and phase switching. At each stage, the model should be judged not by whether it produces a visually plausible pattern, but by whether it reproduces specific measurable features: the sharpness of the written boundary, the spatial homogeneity of the written region, the preferential erasure near interfaces, the persistence of intermediate states after orthogonal pulses, the dependence on surface morphology, and the observed memory effect in which previously written regions are more likely to turn metallic again during later pulses.

The simulation program naturally divides into four parts. The first concerns the ARPES measurement itself and addresses how local orientation and structural variation alter the apparent electronic spectrum. The second concerns spectral mixing and asks whether observed intermediate spectra can be explained as mixtures of known states or require additional intrinsic states. The third concerns pulse-driven transport and heating in the patterned device. The fourth concerns the dynamical evolution of coexisting phases under local driving and relaxation. These parts should not be developed as isolated projects. They must exchange information and constraints. The transport model supplies local fields and temperatures to the phase-evolution model. The phase-evolution model supplies spatially varying states that can be converted into synthetic observables. The ARPES forward model determines how much of the measured spectral variability should be attributed to orientation rather than electronic structure.

## The role of simulation in relation to the experiment

Before describing individual models, it is important to establish what kind of simulation is actually needed. In a system of this kind, there is a temptation to begin with a large finite-element or multiphysics calculation that includes the full sample geometry, anisotropic transport, time-dependent heating, several material parameters, and a phenomenological switching rule. Such a calculation may eventually become useful, but it is not the correct starting point. A large model with many poorly constrained parameters can be made to resemble almost any pattern and will not settle the basic interpretive questions.

The correct approach is hierarchical. One begins by constructing reduced models that answer questions of the following type. If the underlying electronic spectrum were spatially uniform, how much apparent variation would be introduced solely by local changes in the cut through momentum space? If a measured spectrum near a boundary were nothing more than a linear or weakly nonlinear mixture of two endmember spectra, what would that spectrum look like? If heat generation were spatially uniform, would the observed erasing pattern still localize near the interface between phases? If a written state decayed only by ordinary interface retreat, would that be sufficient to explain the spatial memory seen after successive orthogonal pulses? These questions can be answered with relatively compact models. Only after these reduced models have been explored should one proceed to a more comprehensive coupled simulation.

The simulation program should therefore be treated as an instrument for model discrimination. Each model must correspond to a definite physical statement. A model that varies only local orientation asks whether structural artifacts can explain the observed spread in spectra. A two-state mixing model asks whether intermediate spectra are intrinsic or composite. A resistor-thermal network asks whether current crowding and Joule heating naturally create the observed spatial anisotropies. A phase-field or kinetic lattice model asks whether interface energetics and local history are sufficient to generate the boundary-controlled writing and erasing dynamics.

The value of this approach is that even a failed simulation is informative. If a plausible thermal model cannot concentrate erasure near the phase boundary, then pure heating is unlikely to be the whole story. If a local-orientation model can explain most of the pre-pulse spectral diversity, then much of the apparent electronic inhomogeneity is not intrinsic. If a mixture model cannot reproduce the observed line shapes of intermediate regions, then those regions are strong candidates for distinct metastable states. In this sense, the simulations are not auxiliary. They are part of the logic by which the data are interpreted.

## ARPES forward simulation and momentum-space registration

The first simulation that should be built is a forward model of the ARPES measurement under local changes in sample orientation. This is the most immediate route to separating structural effects from electronic ones. The experimental notes indicate that different real-space regions produce spectra whose band centers are shifted in both momentum coordinates, consistent with domains that are slightly tilted away from the nominal surface plane. If that is the case, then two pixels with identical underlying electronic structure may appear to have different spectra simply because the experiment samples different cuts through the Brillouin zone.

The object of this simulation is a latent electronic structure defined in the crystal coordinate system. This may be supplied in several ways. In the simplest version, one may begin with a single representative experimental spectrum or a stitched momentum-energy map taken from a region judged to be structurally uniform. In a more refined version, one may build an interpolated function \(I_0(k_x,k_y,E)\) representing the intensity in the crystal frame. If the material has known symmetry relations, those may be imposed or at least tested.

The forward model then introduces local orientation parameters. A small local tilt changes the mapping between laboratory coordinates and crystal momentum. A local in-plane rotation changes the alignment of the cut relative to high-symmetry directions such as \(\Gamma\)-M. Depending on the geometry of the instrument, local curvature or height variation may also shift the effective emission angles. The essential step is to convert a spectrum defined in the crystal frame into the spectrum that would be measured for a given local orientation. This requires a geometrical transformation from the local crystal axes to the laboratory detection axes, followed by interpolation of the intensity on the transformed grid.

In practice, the model can be implemented in Python using an interpolated data cube. One defines a reference momentum-energy volume, applies local rotation matrices, computes the transformed momentum coordinates sampled by the detector, and interpolates the reference intensity onto those coordinates. Instrumental broadening may be applied afterward by convolution in energy and momentum. If desired, matrix-element modulation can be introduced phenomenologically as a smooth angular weighting function, but the first version should avoid excessive complexity.

This model should first be used on synthetic test cases. One should ask how much a small tilt or rotation changes the apparent positions of the main features, whether it can generate apparent changes in peak separation similar to those seen in the experiment, and whether a quantity such as the fitted \(p3\) position is indeed more sensitive to orientation than a near-\(E_F\) metallicity ratio. Once these tests are complete, the model can be inverted approximately on experimental data. For each real-space point, one may fit local orientation parameters by minimizing the discrepancy between the measured spectrum and the transformed reference spectrum. The resulting field of orientation parameters then becomes a physically interpretable map of local domain geometry.

This step is essential because it produces more than a correction. It provides a new explanatory layer. A boundary identified in the resulting orientation map can be compared directly with the microscope image, with the map of \(p3\)-like quantities, and with the erasing pattern. If many apparent spectral changes collapse after registration into a common Brillouin-zone frame, then structural variation is responsible for much of the apparent heterogeneity. If substantial differences remain after correction, those differences are much more likely to reflect real changes in electronic structure.

The most important output of this simulation is not an image but a registered dataset. Every local spectrum should be resampled into a common momentum-energy coordinate system. Downstream classification, clustering, metallicity mapping, and hypothesis testing should operate primarily on this registered representation rather than on the raw spectra whenever possible. The structural map inferred from the registration should also be saved as an explicit data product, since it will later serve as a predictor in the transport and phase-evolution models.

## Spectral mixing and endmember-based simulation of intermediate states

The second simulation problem concerns intermediate spectra. The experiment already suggests that some regions, especially after erasing pulses, occupy a position between the pristine charge-density-wave state and the current-induced metastable metallic state. The crucial unresolved question is whether these spectra are merely superpositions of two known phases within the finite spatial resolution of the experiment, or whether they are signatures of distinct intermediate states that must be treated as additional electronic configurations.

This question should be attacked by a model that begins from a small number of endmember spectra. One endmember should correspond to a region confidently identified as the equilibrium insulating or charge-density-wave state. A second should correspond to a region confidently identified as the current-written metastable metallic state. If the metallic patch observed before pulsing is spectroscopically distinct, as suggested by the different near-\(E_F\) dispersion character, then it should be treated as a separate endmember rather than folded into the metallic state. If additional robust classes emerge after registration and exploratory analysis, they too may be added.

The simplest model treats a local spectrum as a convex combination of endmember spectra. For a given pixel, one writes the observed intensity as a weighted sum of the endmembers at each momentum and energy. The weights are constrained to be nonnegative and to sum to unity. This linear mixture model is a natural first approximation for spectra produced by unresolved coexistence within the beam footprint or by interpolation across a boundary. The fit can be performed pointwise using constrained least squares or by a probabilistic model with noise terms adapted to the measurement statistics.

This simple model already answers a useful question. If the majority of intermediate spectra can be fitted accurately with only two endmembers, then many apparent intermediate states are likely to be mixed pixels rather than distinct phases. If systematic residuals remain, especially in the neighborhood of the Fermi level or in the dispersion curvature, then a purely mixing-based explanation is insufficient. This is particularly important for determining whether the erased state is just a partial spatial average of metallic and insulating regions or whether it possesses its own intrinsic spectral structure.

A more realistic model may allow slight energy shifts, broadening, or momentum registration errors in each component. This matters because local heating, disorder, or imperfect alignment may alter peak widths and positions without requiring a new phase. One can therefore fit each pixel using endmembers that are allowed to broaden slightly, shift slightly in energy, or deform by low-rank corrections. Such a model is still interpretable while being less brittle than exact linear unmixing.

A still more refined approach is to construct the endmembers not by hand but through a constrained matrix factorization or archetypal decomposition of a carefully curated subset of spectra. Here the important point is not the choice of algorithm but the interpretability of the components. One should not allow a factorization method to absorb arbitrary momentum misregistration and boundary mixing into abstract latent factors. The endmembers must correspond to recognizable physical states or measurement distortions. For this reason, the output of the ARPES registration model should be used before or jointly with any factorization procedure.

Once fitted, the mixture model produces spatial maps of component weights. These maps can be compared with known boundaries, with the written and erased regions, and with pulse history. If a nominally intermediate region is explained almost entirely by a smooth spatial interpolation between insulating and metallic weights, it is likely a coexistence region or a boundary-blurred region. If instead it requires a distinct component with systematic spatial behavior and pulse dependence, that is evidence for a genuine intermediate state.

The synthetic side of this model is equally useful. One can generate expected spectra for hypothetical unresolved phase coexistence at various length scales and compare them with the measured line shapes. This gives a benchmark for what finite spatial resolution alone should produce. If the observed spectra differ qualitatively from these synthetic mixtures, then the experiment is not simply seeing blurred coexistence.

## Transport and thermal simulation on the experimental device geometry

The third and most immediately relevant simulation concerns current flow and Joule heating during the pulse. The experimental observations strongly suggest that the writing and erasing behavior cannot be understood without considering the spatial distribution of current and temperature. The key issue is not merely whether heating occurs, but where heat is generated, how it diffuses, how strongly these distributions depend on the local electronic state and morphology, and whether they naturally produce sharply bounded written regions and interface-focused erasing.

The correct starting point is not a full continuum multiphysics model but a spatially resolved resistor-thermal network constructed on the actual or approximate geometry of the device. The model domain should represent the current bar, contacts, and the measured scan window. The conductivity at each cell should depend on the local state of the material, at minimum distinguishing insulating, metastable metallic, and any relevant intermediate or anomalous states. Additional modifiers may encode morphology inferred from the microscope image or from ARPES-derived orientation maps, for example reduced intercell coupling across step edges or differently oriented flakes.

Current flow may then be computed by solving a discrete form of charge conservation. One imposes a voltage difference or current constraint between the relevant contacts and solves for the potential field on the grid. From the local current density or discrete bond currents, one computes Joule heating. The simplest expression is proportional to \(J^2 \rho\) or, in a resistor-network picture, to \(I^2R\) on each bond or cell. This heating acts as the source term for a thermal diffusion equation defined on the same grid.

The thermal model should include at minimum lateral diffusion within the sample and an effective cooling term representing coupling to the substrate, contacts, and environment. If the pulse duration is short compared with full thermal equilibration, the model must be time dependent. One then computes the temperature field during and immediately after the pulse. The important outputs are the peak temperature reached at each location, the time spent above relevant thresholds, and the cooling rate. This last quantity is especially important because the experimental discussion suggests that not only the maximum temperature but also the dwell time in a certain range may control erasing and recovery.

This model is capable of addressing several essential questions. If the initial state of the sample is spatially uniform, does the device geometry alone produce a broad written bar aligned with the pulse direction? If a metastable metallic bar already exists from a previous pulse, does the orthogonal pulse preferentially concentrate heating at its boundaries? Do morphology-derived boundaries, such as steps or flakes with reduced transverse thermal coupling, create hot spots aligned with the grid-like erased regions? Can an adjacent annealed region arise naturally from heat diffusion without sufficient local drive for switching? These are precisely the kinds of questions that a reduced transport-thermal model can answer before any detailed microscopic switching law is introduced.

The model should remain as close as possible to experimentally observable quantities. Conductivity values can initially be relative rather than absolute. What matters first is the contrast between phases and the effect of boundaries. Later, if resistance measurements and approximate device dimensions are available, the model may be calibrated more quantitatively. The code should therefore be designed so that material parameters, contact conditions, morphology masks, and pulse profiles can be changed without altering the core solver.

The implementation can be carried out in several ways. For a first version, a finite-difference or graph-based solver in Python is adequate and has the advantage of transparency and easy integration with the data-analysis pipeline. Sparse linear algebra libraries are sufficient for the potential solve. Time-dependent thermal diffusion can also be handled on a grid with sparse operators. If later versions require more detailed geometry or anisotropic material tensors, one may migrate selected parts to finite-element packages, but the conceptual framework should remain the same.

The most important output from this model is not a single temperature map but a collection of local driving descriptors. These include current density, dissipated power density, peak temperature, thermal dwell time above threshold, cooling rate, and possibly local temperature gradient. These quantities should be exported in the same spatial coordinate system used for the experimental maps so that they can later be compared directly with the observed writing and erasing patterns and fed into the phase-evolution model.

## Dynamical simulation of phase evolution under pulse-driven forcing

A transport-thermal model alone is not enough, because it does not specify how the local state changes in response to the applied pulse. For that, one needs a phase-evolution model. This model should be designed to test whether the observed spatial dynamics follow naturally from threshold-driven switching, from interface-controlled kinetics, from local memory of previous pulses, or from a combination of these effects.

The state variable in such a model can be defined at several levels of resolution. In the simplest form, each spatial cell may be assigned one of a small number of discrete states, such as insulating charge-density-wave, metastable metallic, and intermediate. The state changes according to transition rules whose rates depend on local drivers such as temperature, current density, or electric field, and on the states of neighboring cells. Such a model is conceptually close to a kinetic Ising or cellular automaton framework, though its transition rules must be chosen to reflect the experimental physics rather than abstract spin dynamics.

A more continuous description introduces an order parameter field that measures the local degree of metallicity or the local amplitude of the charge-density-wave order. The free-energy landscape then contains multiple local minima corresponding to competing states. The pulse modifies this landscape either directly or indirectly through temperature and current-induced terms. The subsequent dynamics are then governed by a relaxation equation, possibly including gradient terms that penalize sharp variations and therefore generate interfaces of finite energy. This is the logic of a phase-field model.

The discrete and continuous descriptions serve similar purposes, but they emphasize different aspects. A discrete-state kinetic model is easier to interpret when the question is whether local history and neighborhood state are enough to explain the observed memory effect. A phase-field model is more natural when one wishes to describe interface motion, barrier lowering, and the distinction between nucleation and growth. In practice, it is often useful to begin with a discrete kinetic model and then move to a continuous one if the data demand more detailed interface physics.

The driving terms for this phase-evolution model should come from the transport-thermal simulation. A cell may be allowed to switch from insulating to metastable only if the local drive exceeds a threshold and if the cooling rate is sufficiently high. A reverse transition may occur if the local temperature enters a range where the metastable state becomes short lived. The transition probability may also depend on whether the cell borders a region already in the alternative phase, which captures the experimental intuition that interfaces play a special role in erasing. If the memory effect is to be modeled, one may include an internal variable representing the remnant susceptibility of a cell to return to a previously visited state. This variable can decay slowly in time or be pinned to structural features.

Such a model allows direct tests of the main experimental patterns. One can ask whether a written metallic bar naturally forms as a homogeneous region with a sharp boundary when the local drive exceeds threshold only within the current path. One can ask whether erasing under an orthogonal pulse localizes near the previous phase boundary, whether repeated pulses along the same direction enlarge the metallic region without strongly changing already metallic interior regions, and whether repeated orthogonal pulsing leads to saturation and the creation of intermediate islands. Because this model evolves a spatial state rather than only a temperature field, it also makes it possible to compare not only where the system is hot, but where it actually changes phase.

The phase-evolution simulation should be designed so that different physical hypotheses correspond to different choices of transition law rather than to different code bases. One version of the transition rule may depend only on peak temperature. Another may depend on both temperature and cooling rate. Another may privilege interfaces by reducing barriers there. Another may incorporate memory of previous pulses. These should all be encoded as interchangeable model variants operating on the same data structures. That makes model comparison possible and prevents the proliferation of ad hoc scripts.

## Coupling simulation outputs back to observables

A simulation of internal state evolution is most useful when it can be translated back into synthetic observables resembling the experiment. The project should therefore include a forward-observation layer. If the phase-evolution model predicts a map of local states, and the spectral mixing model supplies representative spectra for each state, then one can generate a synthetic ARPES map by assigning a local spectrum to each cell and, if necessary, mixing or blurring spectra according to the instrument resolution. If the ARPES registration model supplies local structural orientation, that too can be applied to convert latent local spectra into synthetic measured spectra.

This forward step matters because the experiment does not observe the underlying state field directly. It observes spectra, derived quantities such as \(I_{\mathrm{rat}}\), fitted peak positions, and spatial patterns in those derived maps. A model should therefore be evaluated partly on its ability to reproduce these derived observables rather than only the hidden state map. A written region that looks homogeneous in the latent state variable may not look homogeneous in the measured intensity if structural orientation varies strongly. Conversely, an intrinsically sharp interface may appear broadened by the beam size or by spectral mixing. Without this forward step, one risks comparing a latent simulation variable directly to a measured quantity in a misleading way.

The same principle applies to transport. If the simulation predicts a conducting channel along a pulse direction, that can be converted into an effective resistance along each axis and compared qualitatively or quantitatively with the measured transport history. The experiment suggests that transport and surface ARPES are related but not trivially proportional. The simulation framework should therefore allow both observables to be computed from the same evolving state field. A model that reproduces only the ARPES pattern but fails badly in transport, or vice versa, should be treated cautiously.

## Implementation strategy and numerical workflow

The numerical implementation should be modular from the beginning. The project is likely to evolve as new data become available, and the models will need to be updated repeatedly. A single monolithic notebook is therefore not appropriate. The code should instead be organized into a small research codebase with clear separation between data ingestion, model definitions, parameter files, experiment configuration, and output analysis.

The workflow should begin with data preparation. Experimental maps, microscope images, pulse metadata, and any derived registered spectra should be converted into a stable internal format. Spatial coordinates, pulse sequence labels, and derived masks should all be expressed in a common coordinate convention. Once this layer is in place, the ARPES forward and registration module can operate on the spectral cubes, the mixing module can operate on registered spectra, and the transport-phase modules can operate on spatial grids linked to the same device coordinates.

Each simulation should be executable through a configuration file rather than through manual modification of code. The configuration should define which model variant is being run, which input dataset is being used, what pulse sequence is applied, what material parameters are assumed, and where outputs should be written. This is especially important for later comparison across parameter sweeps or across hypotheses. A simulation result that cannot be traced back to a well-defined set of assumptions is of little scientific value.

The numerical outputs should be saved in machine-readable formats that preserve metadata. For multidimensional arrays and maps, NetCDF, HDF5, or Zarr are appropriate. For scalar summaries or fit results, structured tabular files are sufficient. Static images should never be treated as the primary output of a simulation; they are presentations of underlying data products that must remain accessible for later analysis.

The structure of the files should reflect the structure of the scientific questions. There should be a place where registered spectra live, a place where endmember models live, a place where transport inputs and outputs live, and a place where phase-evolution results and synthetic observables live. Intermediate products, such as morphology masks or interface-distance maps, should also be stored explicitly rather than regenerated implicitly in many different scripts. This matters because such intermediate fields often become scientifically meaningful in their own right.

## Suggested project structure

A useful directory structure for this effort is one in which the code and the data products are clearly separated. The source tree should contain the reusable computational machinery. The data tree should contain raw experimental files, processed datasets, simulation inputs, simulation outputs, and reports. The purpose of this separation is reproducibility. It should always be possible to identify which code version generated which result.

The source directory should contain modules for spectral registration, spectral mixing, transport, thermal evolution, phase evolution, and synthetic observables. A dedicated module for shared utilities should contain interpolation helpers, coordinate transforms, numerical kernels, and I/O functions. A separate configurations directory should contain human-readable parameter files specifying model variants and experiment setups.

The data directory should begin with the raw experimental files, including the original spectral maps and microscope images. A processed directory should then contain registered spectral cubes, extracted endmember spectra, masks of structural boundaries, orientation maps, and derived scalar observables such as \(I_{\mathrm{rat}}\), fitted peak positions, and distance-to-interface fields. A simulation-inputs directory should contain the spatial grids, conductivity maps, initial state maps, pulse definitions, and any geometry masks used by the transport and phase-evolution models. A simulation-results directory should then contain outputs grouped by model family and by run identifier.

A reports or notebooks directory may contain exploratory notebooks and figure-generation scripts, but these should operate on the processed data and simulation outputs rather than performing core numerical work inline. The core solvers should remain in reusable modules so that they can be tested and invoked consistently.

A representative layout may be written as follows.

```text
project_root/
├── README.md
├── pyproject.toml
├── src/
│   └── tase2_switching/
│       ├── io/
│       │   ├── loaders.py
│       │   ├── writers.py
│       │   └── metadata.py
│       ├── arpes/
│       │   ├── registration.py
│       │   ├── forward_model.py
│       │   ├── geometry.py
│       │   └── observables.py
│       ├── spectra/
│       │   ├── endmembers.py
│       │   ├── mixing.py
│       │   ├── decomposition.py
│       │   └── residuals.py
│       ├── transport/
│       │   ├── resistor_network.py
│       │   ├── boundary_conditions.py
│       │   └── conductivity_models.py
│       ├── thermal/
│       │   ├── diffusion.py
│       │   ├── pulse_profiles.py
│       │   └── cooling_models.py
│       ├── phase/
│       │   ├── kinetic_model.py
│       │   ├── phase_field.py
│       │   ├── transition_rules.py
│       │   └── memory.py
│       ├── synthetic/
│       │   ├── spectra.py
│       │   ├── maps.py
│       │   ├── transport_obs.py
│       │   └── beam_blur.py
│       ├── analysis/
│       │   ├── metrics.py
│       │   ├── comparison.py
│       │   └── hypothesis_scores.py
│       └── utils/
│           ├── grids.py
│           ├── interpolation.py
│           ├── plotting.py
│           └── math.py
├── configs/
│   ├── registration/
│   ├── mixing/
│   ├── transport/
│   ├── thermal/
│   ├── phase/
│   └── full_runs/
├── data/
│   ├── raw/
│   │   ├── arpes/
│   │   ├── microscope/
│   │   └── pulse_metadata/
│   ├── processed/
│   │   ├── registered_spectra/
│   │   ├── endmembers/
│   │   ├── orientation_maps/
│   │   ├── structural_masks/
│   │   ├── electronic_state_maps/
│   │   └── derived_observables/
│   ├── simulation_inputs/
│   │   ├── grids/
│   │   ├── initial_conditions/
│   │   ├── conductivity_maps/
│   │   ├── thermal_params/
│   │   └── pulse_sequences/
│   └── simulation_results/
│       ├── registration/
│       ├── mixing/
│       ├── transport/
│       ├── thermal/
│       ├── phase/
│       └── coupled_runs/
├── notebooks/
├── scripts/
│   ├── run_registration.py
│   ├── fit_mixtures.py
│   ├── solve_transport.py
│   ├── solve_thermal.py
│   ├── evolve_phase.py
│   └── run_coupled_model.py
└── reports/
```

This arrangement has several advantages. The conceptual distinction between spectral, transport, thermal, and phase modules is mirrored in the code. Processed data products are preserved and can be reused across many runs. Full coupled runs can be reproduced from configuration files. New model variants can be introduced by adding modules without rewriting the entire workflow.

## Input and output structure for each model family

The ARPES registration model should take as input a spectral cube or a collection of local spectra together with detector coordinate information and, if available, a reference spectrum or latent intensity volume. It should output the estimated local orientation parameters, goodness-of-fit statistics, and the registered spectra in a common coordinate frame. It is advisable to save both the transformed spectra and the transformation parameters. The latter are physically informative and will later be compared with structural boundaries and erase patterns.

The spectral mixing model should take as input the registered spectra and a set of endmember spectra. It should output fitted component weights, residual norms, possibly local broadening or shift parameters, and synthetic reconstructed spectra. Residual maps should always be saved. They are often more informative than the fitted weights because they reveal where the model systematically fails.

The transport-thermal model should take as input a spatial grid, contact definitions, conductivity fields, boundary masks, thermal parameters, and a pulse profile. It should output potential maps, current-density maps, power-dissipation maps, time-dependent temperature fields or compressed descriptors thereof, and effective transport observables such as directional resistance. If possible, the code should save intermediate fields during the pulse, not merely final temperatures, because the time profile may control switching.

The phase-evolution model should take as input an initial state map together with the local driving descriptors generated by the transport-thermal step. It should output time-resolved state maps, interface maps, region labels, memory variables if used, and synthetic observables derived from the final or intermediate states. Since the experiment involves pulse sequences, it is important that the output of one pulse can serve as the initial condition for the next. The data structure should therefore explicitly support chained runs.

The synthetic-observation layer should take simulated state maps, local spectra, structural orientation fields, and instrument-resolution parameters, and should output synthetic spectral cubes, derived scalar maps such as \(I_{\mathrm{rat}}\), fitted peak maps, and transport summaries. These outputs are what allow direct comparison to the experiment.

## Parameter handling, calibration, and comparison across runs

A common failure mode in projects of this kind is the uncontrolled growth of parameters. Conductivities, diffusivities, thresholds, interface energies, memory times, contact resistances, cooling coefficients, and spectral broadening parameters can proliferate quickly. The code structure should therefore distinguish clearly between fixed inputs, physically motivated fitted parameters, and deliberately varied hypothesis parameters.

Some parameters can be estimated from the experiment or from the literature. Others will remain only phenomenological. That is acceptable, provided the distinction is explicit. Each run should therefore carry a metadata file recording the parameter values, their provenance, and the role they play. A good practice is to include a short textual description with each configuration file explaining the physical meaning of the chosen model variant. This turns the simulation archive into a record of hypotheses rather than merely a directory of anonymous numerical runs.

Comparison across runs should be automated. The project should include a comparison layer that reads simulation outputs and computes standardized metrics against the experiment. These metrics may include boundary sharpness, spatial overlap with written and erased regions, correlation with structural masks, reproduction of interface-focused erasing, spectrum reconstruction error, and transport trends across pulse sequences. The purpose of this layer is not to compress the entire science into one scalar score, but to ensure that model comparison is systematic and reproducible.

## Development order

The order in which the simulations should be developed matters. The first stage should be the ARPES registration model, because it cleans and interprets the spectra that all later stages rely upon. The second stage should be the endmember and mixing model, because it determines whether intermediate states can be treated as mixtures or require distinct labels. The third stage should be the resistor-thermal model, beginning with simple geometry and relative conductivities. The fourth stage should be the phase-evolution model, initially with simple threshold rules and later with more elaborate interface and memory terms. Only after these modules work independently should they be coupled into a full synthetic pipeline that produces observables directly comparable to the experiment.

This development order reflects the logic of the science. The first question is what in the data is structural and what is electronic. The second is whether the observed spectra imply more than two electronic states. The third is what physical drive the pulse produces in space and time. The fourth is how the local state evolves under that drive. A comprehensive model that ignores this order is likely to be numerically impressive but conceptually uninformative.

## Scientific value of the simulation program

The simulation effort described here is not merely a tool for generating attractive figures or for post hoc rationalization. If built carefully, it becomes the mechanism by which the experiment can distinguish among qualitatively different interpretations of the data. A forward ARPES model can show whether much of the apparent heterogeneity is geometric. A mixing model can determine whether intermediate spectra are composite or intrinsic. A transport-thermal model can identify whether writing and erasing are consistent with spatially patterned Joule heating and heat dissipation. A phase-evolution model can reveal whether interface-controlled kinetics and local memory are sufficient to explain the observed history dependence.

The most important principle is that each simulation should correspond to a clear physical claim and should produce outputs that can be confronted directly with measured observables. When that principle is respected, the simulation framework becomes part of the argument of the paper rather than a detached numerical appendix. It provides a disciplined way to ask not only whether a mechanism could in principle occur, but whether it is capable of producing the actual spatial and spectral signatures seen in the TaSe\(_2\) current-pulse experiment.
