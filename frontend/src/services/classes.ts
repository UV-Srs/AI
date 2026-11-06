export interface PPEClassConfig {
  id: number;
  key: string;
  label: string;
  color: string;
  borderColor: string;
  bgColor: string;
  textColor: string;
}

export const PPE_CLASSES: Record<string, PPEClassConfig> = {
  person: {
    id: 0,
    key: 'person',
    label: 'Person / Worker',
    color: '#3B82F6',
    borderColor: 'border-blue-500',
    bgColor: 'bg-blue-500/10',
    textColor: 'text-blue-400'
  },
  helmet: {
    id: 1,
    key: 'helmet',
    label: 'Hard Hat / Helmet',
    color: '#F59E0B',
    borderColor: 'border-amber-500',
    bgColor: 'bg-amber-500/10',
    textColor: 'text-amber-400'
  },
  safety_vest: {
    id: 2,
    key: 'safety_vest',
    label: 'Hi-Vis Safety Vest',
    color: '#F97316',
    borderColor: 'border-orange-500',
    bgColor: 'bg-orange-500/10',
    textColor: 'text-orange-400'
  },
  gloves: {
    id: 3,
    key: 'gloves',
    label: 'Safety Gloves',
    color: '#10B981',
    borderColor: 'border-emerald-500',
    bgColor: 'bg-emerald-500/10',
    textColor: 'text-emerald-400'
  },
  safety_shoes: {
    id: 4,
    key: 'safety_shoes',
    label: 'Safety Boots / Shoes',
    color: '#8B5CF6',
    borderColor: 'border-purple-500',
    bgColor: 'bg-purple-500/10',
    textColor: 'text-purple-400'
  },
  goggles: {
    id: 5,
    key: 'goggles',
    label: 'Eye Goggles / Glasses',
    color: '#06B6D4',
    borderColor: 'border-cyan-500',
    bgColor: 'bg-cyan-500/10',
    textColor: 'text-cyan-400'
  },
  mask: {
    id: 6,
    key: 'mask',
    label: 'Face Mask / Respirator',
    color: '#EC4899',
    borderColor: 'border-pink-500',
    bgColor: 'bg-pink-500/10',
    textColor: 'text-pink-400'
  },
  ear_protection: {
    id: 7,
    key: 'ear_protection',
    label: 'Ear Protection / Muffs',
    color: '#EAB308',
    borderColor: 'border-yellow-500',
    bgColor: 'bg-yellow-500/10',
    textColor: 'text-yellow-400'
  }
};

export const ORDERED_CLASS_KEYS = [
  'person',
  'helmet',
  'safety_vest',
  'gloves',
  'safety_shoes',
  'goggles',
  'mask',
  'ear_protection'
];

