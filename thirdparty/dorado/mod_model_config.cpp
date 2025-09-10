#include "toml.h"
#include "error.h"
#include "model_config.h"

#include <unordered_map>


RefinementParams::RefinementParams(int center_idx_)
        : do_rough_rescale(true), center_idx(static_cast<size_t>(center_idx_)) {
    if (center_idx_ < 0) {
        throw std::runtime_error(ERR_STR + "refinement params: 'negative center index'.");
    }
}

RefinementParams parse_refinement_params(const toml::value& config_toml) {
    if (!config_toml.contains("refinement")) {
        return RefinementParams{};
    }

    const auto segment = toml::find(config_toml, "refinement");

    bool do_rough_rescale = toml::find<int>(segment, "refine_do_rough_rescale") == 1;
    if (!do_rough_rescale) {
        return RefinementParams{};
    }

    const int center_index = get_int_in_range(segment, "refine_kmer_center_idx", 0, 19, REQUIRED);
    return RefinementParams(center_index);
}

ModBaseModelConfig::ModBaseModelConfig(const char *model_path_,
                                       ModelGeneralParams general_,
                                       ModificationParams mods_,
                                       ContextParams context_,
                                       RefinementParams refine_)
        : model_path(std::move(model_path_)),
          general(std::move(general_)),
          mods(std::move(mods_)),
          context(general_.model_type == ModelType::CONV_LSTM_V2
                          ? context_.normalised(general.stride)
                          : std::move(context_)),
          refine(std::move(refine_)) {
    // Kmer length is duplicated in modbase model configs - check they match
    if (general.kmer_len != context.kmer_len) {
        auto kl_a = std::to_string(general.kmer_len);
        auto kl_b = std::to_string(context.kmer_len);
        throw std::runtime_error(ERR_STR + "config: 'inconsistent kmer_len: " + kl_a +
                                 " != " + kl_b + "'.");
    }
}

ModBaseModelConfig load_modbase_model_config(const const char *model_path) {
    const auto config_toml = toml::parse(model_path / "config.toml");

    return ModBaseModelConfig{
            model_path, parse_general_params(config_toml), parse_modification_params(config_toml),
            parse_context_params(config_toml), parse_refinement_params(config_toml)};
}
