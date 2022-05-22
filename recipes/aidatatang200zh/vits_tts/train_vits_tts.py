import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


if __name__ == '__main__':
    # output_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath("D:/audiodata/coqui1/")

    # init configs
    dataset_config = BaseDatasetConfig(
        name="aidatatang_200zh",
        meta_file_train="transcript/aidatatang_200_zh_transcript.txt",
        path=os.path.join(output_path, "../aidatatang_200zh/"),
    )

    audio_config = BaseAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        preemphasis=0.0,
        ref_level_db=20,
        log_func="np.log",
        do_trim_silence=False,
        trim_db=23.0,
        mel_fmin=0,
        mel_fmax=None,
        spec_gain=1.0,
        signal_norm=True,
        do_amp_to_db_linear=False,
        resample=False,
    )

    vitsArgs = VitsArgs(
        use_speaker_embedding=True,
        use_sdp=False,
    )

    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name="vits_aidatatang",
        use_speaker_embedding=True,
        batch_size=20,
        eval_batch_size=16,
        batch_group_size=0,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="chinese_mandarin_cleaners",
        use_phonemes=True,
        phoneme_language="zh-cn",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        precompute_num_workers=8,
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=False,
        mixed_precision=True,
        sort_by_audio_len=False,
        min_audio_len=256 * 4,
        max_audio_len=500000,
        output_path=output_path,
        datasets=dataset_config,
        test_sentences=[
            "测试一下声音，中文试试.",
        ],
        characters=CharactersConfig(
            pad="_",
            eos="~",
            bos="^",
            blank=None,
            characters="",
            # will be replaced by phonemes if use_phonemes is true
            punctuations="\uff0c\u3002\uff1f\uff01\uff5e\uff1a\uff1b*\u2014\u2014-\uff08\uff09\u3010\u3011!'(),-.:;? ",
            phonemes="ABCDEFGHIJKLMNOPQRSTUVWXYZ12345giy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b",
            is_unique=False,
            is_sorted=True
        ),
    )

    ## INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init speaker manager for multi-speaker training
    # it maps speaker-id to speaker-name in the model and data-loader
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()
