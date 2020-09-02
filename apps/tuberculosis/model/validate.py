from autumn.tool_kit import schema_builder as sb

validate_params = sb.build_validator(
    # Country info
    iso3=str,
    region=sb.Nullable(str),
    # Running time.
    start_time=float,
    end_time=float,
    time_step=float,
    # base disease model
    contact_rate=float,
    override_latency_rates=bool,
    stabilisation_rate=float,
    late_activation_rate=float,
    early_activation_rate=float,
    detection_rate=float,
    treatment_recovery_rate=float,
    relapse_rate=float,
    treatment_death_rate=float,
    self_recovery_rate_dict=dict,
    infect_death_rate_dict=dict,
    rr_infection_latent=float,
    rr_infection_recovered=float,
    # stratification requests
    stratify_by=list,
    # defining organ stratification
    incidence_props_pulmonary=float,
    incidence_props_smear_positive_among_pulmonary=float,
    smear_negative_infect_multiplier=float,
    extrapulmonary_infect_multiplier=float,
)
