// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

const isPublic = process.env.DEPLOY_TARGET === 'public';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'TiRex',
  favicon: 'img/favicon.ico',
  url: isPublic ? 'https://nx-ai.github.io' : 'http://localhost:3000',
  baseUrl: isPublic ? '/tirex/' : '/',
  organizationName: 'NX-AI',
  projectName: isPublic ? 'tirex' : 'tirex-internal',
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  trailingSlash: false,
  i18n: { defaultLocale: 'en', locales: ['en'] },
  staticDirectories: ['static'],

  presets: [
    [
      'classic',
      ({
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/NX-AI/tirex-internal/edit/main/',
        },
        blog: false,
        theme: { customCss: require.resolve('./src/css/custom.css') },
      })
    ]
  ],

  themeConfig: ({
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false
    },
    navbar: {
      title: 'TiRex Documentation',
      logo: { alt: 'NXAI Logo', src: 'img/NXAI-Logo.png', srcDark: 'img/NXAI-Logo-white.png' },
      items: [
        {type: 'docSidebar', sidebarId: 'docsSidebar', position: 'left', label: 'Docs'},
        {
          href: 'https://github.com/NX-AI/tirex',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
        {
          href: 'https://huggingface.co/NX-AI/TiRex',
          position: 'right',
          className: 'header-hf-link',
          'aria-label': 'Hugging Face model card',
        },
        {
          href: 'https://arxiv.org/abs/2505.23719',
          position: 'right',
          className: 'header-arxiv-link',
          'aria-label': 'Paper (arXiv)',
        },
      ]
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash']
    }
  })
};
export default config;
