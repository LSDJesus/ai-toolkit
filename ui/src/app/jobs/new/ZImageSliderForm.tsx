'use client';

import { JobConfig, ZImageSliderTarget } from '@/types';
import { TextInput, SelectInput, NumberInput, Checkbox } from '@/components/formInputs';
import Card from '@/components/Card';
import { Plus, Trash2 } from 'lucide-react';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
};

export default function ZImageSliderForm({ jobConfig, setJobConfig }: Props) {
  const zimage = jobConfig.config.process[0].zimage_slider;
  if (!zimage) return null;

  const addTarget = () => {
    const newTargets = [
      ...(zimage.targets || []),
      { positive: '', negative: '', target_class: 'a person, portrait photo', weight: 1.0 },
    ];
    setJobConfig(newTargets, 'config.process[0].zimage_slider.targets');
  };

  const removeTarget = (index: number) => {
    const newTargets = zimage.targets.filter((_: any, i: number) => i !== index);
    setJobConfig(newTargets, 'config.process[0].zimage_slider.targets');
  };

  const addSamplePrompt = () => {
    const newPrompts = [...(zimage.sample_prompts || []), ''];
    setJobConfig(newPrompts, 'config.process[0].zimage_slider.sample_prompts');
  };

  const removeSamplePrompt = (index: number) => {
    const newPrompts = zimage.sample_prompts.filter((_: any, i: number) => i !== index);
    setJobConfig(newPrompts, 'config.process[0].zimage_slider.sample_prompts');
  };

  return (
    <div className="space-y-6">
      {/* Model & Encoder */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card title="Z-Image Model">
          <TextInput
            label="DiT Model Path"
            value={zimage.model_path}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.model_path')}
            placeholder="Path to Z-Image .safetensors"
            required
          />
          <Checkbox
            label="Is Turbo Model"
            checked={zimage.is_turbo}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.is_turbo')}
          />
          <TextInput
            label="VAE Path"
            value={zimage.vae_path}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.vae_path')}
            placeholder="Path to FLUX ae.safetensors"
          />
        </Card>

        <Card title="Text Encoder">
          <SelectInput
            label="Encoder Type"
            value={zimage.encoder_type}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.encoder_type')}
            options={[
              { value: 'gguf', label: 'GGUF (Q4_K_M, lightweight)' },
              { value: 'safetensors', label: 'Safetensors (full precision)' },
            ]}
          />
          <TextInput
            label="Encoder Path"
            value={zimage.encoder_path}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.encoder_path')}
            placeholder="Path to encoder model"
            required
          />
          {zimage.encoder_type === 'gguf' && (
            <>
              <TextInput
                label="Projector Path"
                value={zimage.encoder_projector_path}
                onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.encoder_projector_path')}
                placeholder="mmproj GGUF path"
              />
              <TextInput
                label="VL-to-Base Adapter"
                value={zimage.encoder_adapter_path}
                onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.encoder_adapter_path')}
                placeholder="models/vl_adapter/vl_to_base_adapter_best.pt"
              />
            </>
          )}
          <SelectInput
            label="Encoder Device"
            value={zimage.encoder_device}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.encoder_device')}
            options={[
              { value: 'cuda:0', label: 'GPU 0' },
              { value: 'cuda:1', label: 'GPU 1' },
              { value: 'cpu', label: 'CPU' },
            ]}
          />
        </Card>

        <Card title="LoRA Config">
          <NumberInput
            label="Rank"
            value={zimage.rank}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.rank')}
            min={1}
            max={256}
          />
          <NumberInput
            label="Alpha"
            value={zimage.alpha}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.alpha')}
            min={1}
            max={256}
          />
          <div className="text-xs text-gray-500 mt-1">
            Alpha/Rank ratio = {(zimage.alpha / zimage.rank).toFixed(2)}× per weight unit.
            Lower ratio = finer control at extreme weights.
          </div>
        </Card>
      </div>

      {/* Training Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card title="Training">
          <NumberInput
            label="Steps"
            value={zimage.steps}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.steps')}
            min={1}
          />
          <NumberInput
            label="Learning Rate"
            value={zimage.lr}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.lr')}
            min={0}
          />
          <NumberInput
            label="Weight Decay"
            value={zimage.weight_decay}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.weight_decay')}
            min={0}
          />
          <Checkbox
            label="Gradient Checkpointing"
            checked={zimage.gradient_checkpointing}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.gradient_checkpointing')}
          />
        </Card>

        <Card title="Denoising">
          <NumberInput
            label="Max Denoising Steps"
            value={zimage.max_denoising_steps}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.max_denoising_steps')}
            min={1}
            max={50}
          />
          <NumberInput
            label="Denoise Fraction"
            value={zimage.denoise_to_fraction}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.denoise_to_fraction')}
            min={0.1}
            max={1.0}
          />
          <NumberInput
            label="Sigma Shift"
            value={zimage.sigma_shift}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.sigma_shift')}
            min={0}
          />
          <NumberInput
            label="CFG Scale"
            value={zimage.cfg_scale}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.cfg_scale')}
            min={1}
          />
        </Card>

        <Card title="Save & Log">
          <TextInput
            label="Output Path"
            value={zimage.output_path}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.output_path')}
            placeholder="sliders/slider.safetensors"
          />
          <NumberInput
            label="Save Every N Steps"
            value={zimage.save_every}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.save_every')}
            min={1}
          />
          <NumberInput
            label="Sample Every N Steps"
            value={zimage.sample_every}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.sample_every')}
            min={1}
          />
          <NumberInput
            label="Log Every N Steps"
            value={zimage.log_every}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.log_every')}
            min={1}
          />
        </Card>

        <Card title="Sampling">
          <NumberInput
            label="Sample Steps"
            value={zimage.sample_steps}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.sample_steps')}
            min={1}
            max={100}
          />
          <NumberInput
            label="Seed"
            value={zimage.sample_seed}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.sample_seed')}
            min={0}
          />
          <SelectInput
            label="Device"
            value={zimage.device}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.device')}
            options={[
              { value: 'cuda:0', label: 'GPU 0' },
              { value: 'cuda:1', label: 'GPU 1' },
            ]}
          />
          <SelectInput
            label="Dtype"
            value={zimage.dtype}
            onChange={value => setJobConfig(value, 'config.process[0].zimage_slider.dtype')}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp16', label: 'FP16' },
            ]}
          />
        </Card>
      </div>

      {/* Slider Targets */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Slider Targets</h2>
          <button
            type="button"
            onClick={addTarget}
            className="flex items-center gap-1 px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-sm"
          >
            <Plus size={14} /> Add Target
          </button>
        </div>
        {zimage.targets.map((target: ZImageSliderTarget, idx: number) => (
          <Card key={idx} title={`Target ${idx + 1}`}>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Positive Prompt (older / more)</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm min-h-[120px] resize-y"
                  value={target.positive}
                  onChange={e => setJobConfig(e.target.value, `config.process[0].zimage_slider.targets[${idx}].positive`)}
                  placeholder="Describe the positive direction (e.g., mature woman with laugh lines...)"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Negative Prompt (younger / less)</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm min-h-[120px] resize-y"
                  value={target.negative}
                  onChange={e => setJobConfig(e.target.value, `config.process[0].zimage_slider.targets[${idx}].negative`)}
                  placeholder="Describe the negative direction (e.g., teenage girl with smooth skin...)"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Neutral Anchor</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm min-h-[80px] resize-y"
                  value={target.target_class}
                  onChange={e => setJobConfig(e.target.value, `config.process[0].zimage_slider.targets[${idx}].target_class`)}
                  placeholder="a person, portrait photo"
                />
                <NumberInput
                  label="Loss Weight"
                  value={target.weight}
                  onChange={value => setJobConfig(value, `config.process[0].zimage_slider.targets[${idx}].weight`)}
                  min={0}
                />
                {zimage.targets.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeTarget(idx)}
                    className="flex items-center gap-1 mt-2 px-2 py-1 bg-red-800 hover:bg-red-700 rounded text-xs"
                  >
                    <Trash2 size={12} /> Remove
                  </button>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Sample Prompts */}
      <Card title="Sample Prompts">
        <div className="space-y-2">
          {zimage.sample_prompts.map((prompt: string, idx: number) => (
            <div key={idx} className="flex gap-2">
              <textarea
                className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm min-h-[60px] resize-y"
                value={prompt}
                onChange={e => setJobConfig(e.target.value, `config.process[0].zimage_slider.sample_prompts[${idx}]`)}
                placeholder="Sample prompt for visualization during training..."
              />
              {zimage.sample_prompts.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeSamplePrompt(idx)}
                  className="px-2 bg-red-800 hover:bg-red-700 rounded"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))}
          <button
            type="button"
            onClick={addSamplePrompt}
            className="flex items-center gap-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            <Plus size={14} /> Add Prompt
          </button>
        </div>
        <div className="mt-3 text-xs text-gray-500">
          Sample weights: {zimage.sample_weights.map((w: number) => w.toFixed(1)).join(', ')}
          <span className="ml-2">(images generated at each weight per prompt per sample_every steps)</span>
        </div>
      </Card>
    </div>
  );
}
