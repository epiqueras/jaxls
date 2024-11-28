module.exports = {
    parser: 'typescript',
    singleQuote: true,
    printWidth: 120,
    tabWidth: 4,
    endOfLine: 'auto',
    trailingComma: 'all',
    overrides: [
        {
            files: ['*.yml', '*.yaml'],
            options: {
                parser: 'yaml',
                tabWidth: 2,
            },
        },
    ],
};
