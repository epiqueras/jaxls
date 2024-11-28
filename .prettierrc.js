module.exports = {
  bracketSameLine: true,
  bracketSpacing: true,
  importOrder: [
    "<BUILTIN_MODULES>",
    "<THIRD_PARTY_MODULES>",
    "",
    "^~(.*)$",
    "",
    "^@/(.*)$",
    "",
    "^[./]"
  ],
  plugins: ["@ianvs/prettier-plugin-sort-imports"],
  printWidth: 80,
  semi: false,
  singleQuote: false,
  tabWidth: 2,
  trailingComma: "none",
  useTabs: false
}
