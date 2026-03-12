interface Props {
  value: string;
  onChange: (v: string) => void;
}

export default function TokenInput({ value, onChange }: Props) {
  return (
    <div className="flex items-center gap-2">
      <label htmlFor="token" className="text-xs text-gray-400 whitespace-nowrap">
        Bearer token
      </label>
      <input
        id="token"
        type="password"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="eyJ..."
        className="w-48 px-3 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded-md focus:outline-none focus:border-brand-500 text-gray-200 placeholder-gray-600"
      />
    </div>
  );
}
