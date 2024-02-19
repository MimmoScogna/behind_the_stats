'use strict';(function(){var cmpFile='noModule'in HTMLScriptElement.prototype?'cmp2.js':'cmp2-polyfilled.js';(function(){var cmpScriptElement=document.createElement('script');var firstScript=document.getElementsByTagName('script')[0];cmpScriptElement.async=true;cmpScriptElement.type='text/javascript';var cmpVersion;var tagUrl=document.currentScript.src;cmpVersion='https://cmp.inmobi.com/tcfv2/CMP_FILE?referer=www.whoscored.com'.replace('CMP_FILE',cmpFile);cmpScriptElement.src=cmpVersion;firstScript.parentNode.insertBefore(cmpScriptElement,firstScript);})();(function(){var css=""
+" .qc-cmp-button { "
+"   background-color: #84bf41 !important; "
+"   border-color: #84bf41 !important; "
+" } "
+" .qc-cmp-button:hover { "
+"   border-color: #84bf41 !important; "
+" } "
+" .qc-cmp-alt-action, "
+" .qc-cmp-link { "
+"   color: #84bf41 !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button:hover { "
+"   background-color: #84bf41 !important; "
+"   border-color: #84bf41 !important; "
+" } "
+" .qc-cmp-button { "
+"   color: #ffffff !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button:hover { "
+"   color: #ffffff !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button { "
+"   color: #262626 !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button { "
+"   background-color: #e2e4de !important; "
+"   border-color: transparent !important; "
+" } "
+" .qc-cmp-ui, "
+" .qc-cmp-ui .qc-cmp-title, "
+" .qc-cmp-ui .qc-cmp-table, "
+" .qc-cmp-ui .qc-cmp-messaging, "
+" .qc-cmp-ui .qc-cmp-sub-title, "
+" .qc-cmp-ui .qc-cmp-vendor-list, "
+" .qc-cmp-ui .qc-cmp-purpose-info, "
+" .qc-cmp-ui .qc-cmp-table-header, "
+" .qc-cmp-ui .qc-cmp-beta-messaging, "
+" .qc-cmp-ui .qc-cmp-main-messaging, "
+" .qc-cmp-ui .qc-cmp-vendor-list-title{ "
+"   color: #ffffff !important; "
+" } "
+" .qc-cmp-ui a, "
+" .qc-cmp-ui .qc-cmp-alt-action { "
+"   color: #84bf41 !important; "
+" } "
+" .qc-cmp-ui { "
+"   background-color: #262626 !important; "
+" } "
+" .qc-cmp-small-toggle.qc-cmp-toggle-on, "
+" .qc-cmp-toggle.qc-cmp-toggle-on { "
+"   background-color: #84bf41!important; "
+"   border-color: #84bf41!important; "
+" } "
+".qc-cmp2-persistent-link { z-index: 999999998 !important }"
+""
+"";var stylesElement=document.createElement('style');var re=new RegExp('&quote;','g');css=css.replace(re,'"');stylesElement.type='text/css';if(stylesElement.styleSheet){stylesElement.styleSheet.cssText=css;}else{stylesElement.appendChild(document.createTextNode(css));}
var head=document.head||document.getElementsByTagName('head')[0];head.appendChild(stylesElement);})();var autoDetectedLanguage='en';var gvlVersion=2;function splitLang(lang){return lang.length>2?lang.split('-')[0]:lang;};function isSupported(lang){var langs=['en','fr','de','it','es','da','nl','el','hu','pt','pt-br','pt-pt','ro','fi','pl','sk','sv','no','ru','bg','ca','cs','et','hr','lt','lv','mt','sl','tr','zh'];return langs.indexOf(lang)===-1?false:true;};if(gvlVersion===2&&isSupported(splitLang(document.documentElement.lang))){autoDetectedLanguage=splitLang(document.documentElement.lang);}else if(gvlVersion===3&&isSupported(document.documentElement.lang)){autoDetectedLanguage=document.documentElement.lang;}else if(isSupported(splitLang(navigator.language))){autoDetectedLanguage=splitLang(navigator.language);};var choiceMilliSeconds=(new Date).getTime();window.__tcfapi('init',2,function(){},{"coreConfig":{"uspVersion":1,"uspJurisdiction":["CA"],"uspLspact":"N","suppressCcpaLinks":false,"inmobiAccountId":"01HxE0C8MDYEG","privacyMode":["GDPR","USP"],"hashCode":"Mq3S3sSBe8s/rtQ6lTDQ8w","publisherCountryCode":"GB","publisherName":"whoscored.com","vendorPurposeIds":[2,3,4,5,6,7,8,9,10,1],"vendorFeaturesIds":[1,3,2],"vendorPurposeLegitimateInterestIds":[3,5,7,8,9,2,4,10,6],"vendorSpecialFeaturesIds":[1,2],"vendorSpecialPurposesIds":[1,2],"googleEnabled":true,"consentScope":"service","thirdPartyStorageType":"iframe","consentOnSafari":false,"displayUi":"inEU","defaultToggleValue":"off","initScreenRejectButtonShowing":false,"softOptInEnabled":false,"showSummaryView":true,"persistentConsentLinkLocation":3,"displayPersistentConsentLink":true,"uiLayout":"banner","publisherLogo":"https://d2zywfiolv4f83.cloudfront.net/img/ws_logo.svg","rejectConsentRedirectUrl":"https://www.whoscored.com","vendorListUpdateFreq":30,"publisherPurposeIds":[1,2,3,4,5,6,7,8,9,10],"initScreenBodyTextOption":1,"publisherConsentRestrictionIds":[],"publisherLIRestrictionIds":[],"publisherPurposeLegitimateInterestIds":[],"publisherSpecialPurposesIds":[1,2],"publisherFeaturesIds":[1,2,3],"publisherSpecialFeaturesIds":[1,2],"stacks":[],"lang_":autoDetectedLanguage,"gvlVersion":2},"premiumUiLabels":{},"premiumProperties":{"googleWhitelist":[1]},"coreUiLabels":{},"theme":{"uxPrimaryButtonTextColor":"#ffffff","uxPrimaryButtonColor":"#84bf41","uxBackgroundColor":"#262626","uxSecondaryButtonColor":"#e2e4de","uxSecondaryButtonTextColor":"#262626","uxToogleActiveColor":"#84bf41","uxLinkColor":"#84bf41","uxFontColor":"#ffffff"},"nonIabVendorsInfo":{}});})();