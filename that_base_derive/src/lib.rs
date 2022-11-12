use proc_macro::TokenStream;
use quote::{__private::Span, quote, ToTokens};
use syn::{
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    visit_mut::{visit_lifetime_mut, VisitMut},
    Data, DataEnum, DataStruct, DeriveInput, Field, Fields, GenericParam, Ident, Index, Lifetime,
    Path, Type, Variant, Visibility,
};

#[proc_macro_derive(Datum)]
pub fn datum(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident, generics, ..
    } = parse_macro_input!(input as DeriveInput);
    let (impl_generics, type_generics, where_clauses) = generics.split_for_impl();
    let datum = path(ident.span(), ["that_bass", "Datum"]);
    quote!(
        #[automatically_derived]
        impl #impl_generics #datum for #ident #type_generics #where_clauses { }
    )
    .into()
}

#[proc_macro_derive(Template)]
pub fn template(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        data: Data::Struct(DataStruct { fields, .. }),
        ..
    } = parse_macro_input!(input as DeriveInput) else {
        return quote!(compile_error!("Enumeration and union types are not supported for this derive.");).into();
    };
    let (impl_generics, type_generics, where_clauses) = generics.split_for_impl();
    let template_path = path(ident.span(), ["that_bass", "template", "Template"]);
    let error_path = path(ident.span(), ["that_bass", "Error"]);
    let declare_path = path(ident.span(), ["that_bass", "template", "DeclareContext"]);
    let initialize_path = path(ident.span(), ["that_bass", "template", "InitializeContext"]);
    let apply_path = path(ident.span(), ["that_bass", "template", "ApplyContext"]);
    let (construct, _, names, types, _, indices) = deconstruct_fields(&fields);
    let construct = construct(&names);
    quote!(
        #[automatically_derived]
        unsafe impl #impl_generics #template_path for #ident #type_generics #where_clauses {
            type State = (#(<#types as #template_path>::State,)*);
            fn declare(mut _context: #declare_path) -> Result<(), #error_path> {
                #(<#types as #template_path>::declare(_context.own())?;)*
                Ok(())
            }
            fn initialize(mut _context: #initialize_path) -> Result<Self::State, #error_path> {
                Ok((#(<#types as #template_path>::initialize(_context.own())?,)*))
            }
            #[inline]
            unsafe fn apply(self, _state: &Self::State, mut _context: #apply_path) {
                let #ident #construct = self;
                #(#template_path::apply(#names, &_state.#indices, _context.own());)*
            }
        }
    )
    .into()
}

#[proc_macro_derive(Filter)]
pub fn filter(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);
    let (impl_generics, type_generics, where_clauses) = generics.split_for_impl();
    let filter_path = path(ident.span(), ["that_bass", "filter", "Filter"]);
    let any_path = path(ident.span(), ["that_bass", "filter", "Any"]);
    let context_path = path(ident.span(), ["that_bass", "filter", "Context"]);
    let table_path = path(ident.span(), ["that_bass", "table", "Table"]);
    match data {
        Data::Struct(DataStruct { fields, .. }) => {
            let (construct, _, names, ..) = deconstruct_fields(&fields);
            let construct = construct(&names);
            quote!(
                #[automatically_derived]
                impl #impl_generics #filter_path for #ident #type_generics #where_clauses {
                    fn filter(&self, _table: &#table_path, _context: #context_path) -> bool {
                        let #ident #construct = self;
                        true #(&& #filter_path::filter(#names, _table, _context.clone()))*
                    }
                }

                #[automatically_derived]
                impl #impl_generics #filter_path for #any_path<#ident #type_generics> #where_clauses {
                    fn filter(&self,_table: &#table_path, _context: #context_path) -> bool {
                        let #ident #construct = self.inner();
                        false #(|| #filter_path::filter(#names, _table, _context.clone()))*
                    }
                }
            )
        }
        Data::Enum(DataEnum { variants, .. }) => {
            let (all, any): (Vec<_>, Vec<_>) = variants
                .into_iter()
                .map(|Variant { ident:name, fields, .. }| {
                    let (construct, _, names, ..) = deconstruct_fields(&fields);
                    let construct = construct(&names);
                    (
                        quote!(#ident::#name #construct => true #(&& #filter_path::filter(#names, _table, _context.clone()))*),
                        quote!(#ident::#name #construct => false #(|| #filter_path::filter(#names, _table, _context.clone()))*),
                    )
                })
                .unzip();
            quote!(
                #[automatically_derived]
                impl #impl_generics #filter_path for #ident #type_generics #where_clauses {
                    fn filter(&self, _table: &#table_path, _context: #context_path) -> bool {
                        match self { #(#all,)* _ => true }
                    }
                }

                #[automatically_derived]
                impl #impl_generics #filter_path for #any_path<#ident #type_generics> #where_clauses {
                    fn filter(&self,_table: &#table_path, _context: #context_path) -> bool {
                        match self.inner() { #(#any,)* _ => false }
                    }
                }
            )
        }
        Data::Union(_) => quote!(compile_error!("Union types are not supported for this derive.");),
    }
    .into()
}

#[proc_macro_derive(Row)]
pub fn row(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        data,
        vis,
        ..
    } = parse_macro_input!(input as DeriveInput);

    struct LifetimeVisitor<'a>(&'a str);
    impl VisitMut for LifetimeVisitor<'_> {
        fn visit_lifetime_mut(&mut self, i: &mut Lifetime) {
            i.ident = Ident::new(self.0, i.span());
            visit_lifetime_mut(self, i);
        }
    }
    let (impl_generics, _, where_clauses) = generics.split_for_impl();
    let mut static_generics = generics.clone();
    LifetimeVisitor("static".into()).visit_generics_mut(&mut static_generics);
    let (_, type_generics, _) = static_generics.split_for_impl();

    let row_path = path(ident.span(), ["that_bass", "row", "Row"]);
    let declare_path = path(ident.span(), ["that_bass", "row", "DeclareContext"]);
    let initialize_path = path(ident.span(), ["that_bass", "row", "InitializeContext"]);
    let item_path = path(ident.span(), ["that_bass", "row", "ItemContext"]);
    let chunk_path = path(ident.span(), ["that_bass", "row", "ChunkContext"]);
    let error_path = path(ident.span(), ["that_bass", "Error"]);
    match data {
        Data::Struct(DataStruct { mut fields, .. }) => {
            LifetimeVisitor("static").visit_fields_mut(&mut fields);
            let (construct, define, names, types, visibilities, indices) = deconstruct_fields(&fields);
            let construct = construct(&names);
            let strip_generics: Vec<_> = generics.params.iter()
                .filter_map(|generic| {
                    match generic {
                        GenericParam::Lifetime(_) => None,
                        generic => Some(generic.clone()),
                    }
                })
                .collect();

            let (state_name, state_struct) = {
                let name = Ident::new(&format!("{}__State__", ident), ident.span());
                let types: Vec<Type> = types.iter().map(|ty| parse_quote!(<#ty as #row_path>::State)).collect();
                let define = define(&visibilities, &names, &types);
                (name.clone(), quote!(
                    #[allow(non_camel_case_types)]
                    #vis struct #name #define
                ))
            };
            let (read_name, read_struct) = {
                let name = Ident::new(&format!("{}__Read__", ident), ident.span());
                let types: Vec<Type> = types.iter().map(|ty| parse_quote!(<#ty as #row_path>::Read)).collect();
                let define = define(&visibilities, &names, &types);
                (name.clone(), quote!(
                    #[allow(non_camel_case_types)]
                    #vis struct #name #define
                ))
            };
            let (item_name, item_struct) = {
                let name = Ident::new(&format!("{}__Item__", ident), ident.span());
                let name = quote!(#name<'__item__ #(,#strip_generics)*>);
                let types: Vec<Type> = types.iter().map(|ty| parse_quote!(<#ty as #row_path>::Item<'__item__>)).collect();
                let define = define(&visibilities, &names, &types);
                (name.clone(), quote!(
                    #[allow(non_camel_case_types)]
                    #vis struct #name #define
                ))
            };
            let (chunk_name, chunk_struct) = {
                let name = Ident::new(&format!("{}__Chunk__", ident), ident.span());
                let name = quote!(#name<'__chunk__ #(,#strip_generics)*>);
                let types: Vec<Type> = types.iter().map(|ty| parse_quote!(<#ty as #row_path>::Chunk<'__chunk__>)).collect();
                let define = define(&visibilities, &names, &types);
                (name.clone(), quote!(
                    #[allow(non_camel_case_types)]
                    #vis struct #name #define
                ))
            };

            quote!(
                #state_struct
                #read_struct
                #item_struct
                #chunk_struct

                #[automatically_derived]
                unsafe impl #impl_generics #row_path for #ident #type_generics #where_clauses {
                    type State = #state_name;
                    type Read = #read_name;
                    type Item<'__item__> = #item_name;
                    type Chunk<'__chunk__> = #chunk_name;

                    fn declare(mut _context: #declare_path) -> Result<(), #error_path> {
                        #(<#types as #row_path>::declare(_context.own())?;)*
                        Ok(())
                    }
                    fn initialize(_context: #initialize_path) -> Result<Self::State, #error_path> {
                        #(let #names = <#types as #row_path>::initialize(_context.own())?;)*
                        Ok(Self::State #construct)
                    }
                    fn read(_state: &Self::State) -> <Self::Read as Row>::State {
                        #(let #names = <#types as #row_path>::read(&_state.#indices);)*
                        Self::Read #construct
                    }
                    #[inline]
                    unsafe fn item<'__item__>(_state: &'__item__ Self::State, _context: #item_path<'__item__>) -> Self::Item<'__item__> {
                        #(let #names = <#types as #row_path>::item(&_state.#indices, _context.own());)*
                        Self::Item::<'__item__> #construct
                    }
                    #[inline]
                    unsafe fn chunk<'__chunk__>(_state: &'__chunk__ Self::State, _context: #chunk_path<'__chunk__>) -> Self::Chunk<'__chunk__> {
                        #(let #names = <#types as #row_path>::chunk(&_state.#indices, _context.own());)*
                        Self::Chunk::<'__chunk__> #construct
                    }
                }

                // #[automatically_derived]
                // unsafe impl #impl_generics #row_path for #ident #type_generics #where_clauses {
                //     type State = (#(<#types as #row_path>::State,)*);
                //     type Read = (#(<#types as #row_path>::Read,)*);
                //     type Item<'__item__> = (#(<#types as #row_path>::Item<'__item__>,)*);
                //     type Chunk<'__chunk__> = (#(<#types as #row_path>::Chunk<'__chunk__>,)*);

                //     fn declare(mut _context: #declare_path) -> Result<(), #error_path> {
                //         #(<#types as #row_path>::declare(_context.own())?;)*
                //         Ok(())
                //     }
                //     fn initialize(_context: #initialize_path) -> Result<Self::State, #error_path> {
                //         Ok((#(<#types as #row_path>::initialize(_context.own())?,)*))
                //     }
                //     fn read(_state: &Self::State) -> <Self::Read as Row>::State {
                //         (#(<#types as #row_path>::read(&_state.#indices),)*)
                //     }
                //     #[inline]
                //     unsafe fn item<'__item__>(_state: &'__item__ Self::State, _context: #item_path<'__item__>) -> Self::Item<'__item__> {
                //         (#(<#types as #row_path>::item(&_state.#indices, _context.own()),)*)
                //     }
                //     #[inline]
                //     unsafe fn chunk<'__chunk__>(_state: &'__chunk__ Self::State, _context: #chunk_path<'__chunk__>) -> Self::Chunk<'__chunk__> {
                //         (#(<#types as #row_path>::chunk(&_state.#indices, _context.own()),)*)
                //     }
                // }
            )
        }
        Data::Enum(DataEnum { .. }) => quote!(),
        Data::Union(_) => quote!(compile_error!("Union types are not supported for this derive.");),
    }
    .into()
}

fn path<'a>(span: Span, segments: impl IntoIterator<Item = &'a str>) -> Path {
    let mut separated = Punctuated::new();
    for segment in segments {
        separated.push(Ident::new(segment, span).into());
    }
    Path {
        segments: separated,
        leading_colon: None,
    }
}

fn deconstruct_fields(
    fields: &Fields,
) -> (
    fn(&[Ident]) -> Box<dyn ToTokens>,
    fn(&[Visibility], &[Ident], &[Type]) -> Box<dyn ToTokens>,
    Vec<Ident>,
    Vec<Type>,
    Vec<Visibility>,
    Vec<Index>,
) {
    match fields {
        Fields::Named(fields) => {
            let mut names = Vec::new();
            let mut types = Vec::new();
            let mut visibilities = Vec::new();
            let mut indices = Vec::new();
            for Field { ident, ty, vis, .. } in fields.named.iter() {
                if let Some(ident) = ident {
                    names.push(ident.clone());
                    types.push(ty.clone());
                    visibilities.push(vis.clone());
                    indices.push(Index::from(indices.len()));
                }
            }
            (
                |names| Box::new(quote!({ #(#names,)* })),
                |visibilities, names, types| {
                    Box::new(quote!({ #(#visibilities #names: #types,)* }))
                },
                names,
                types,
                visibilities,
                indices,
            )
        }
        Fields::Unnamed(fields) => {
            let mut names = Vec::new();
            let mut types = Vec::new();
            let mut visibilities = Vec::new();
            let mut indices = Vec::new();
            for Field { ty, vis, .. } in fields.unnamed.iter() {
                names.push(Ident::new(&format!("_{}", names.len()), ty.span()));
                types.push(ty.clone());
                visibilities.push(vis.clone());
                indices.push(Index::from(indices.len()));
            }
            (
                |names| Box::new(quote!((#(#names,)*))),
                |visibilities, _, types| Box::new(quote!((#(#visibilities #types,)*))),
                names,
                types,
                visibilities,
                indices,
            )
        }
        Fields::Unit => (
            |_| Box::new(quote!()),
            |_, _, _| Box::new(quote!()),
            vec![],
            vec![],
            vec![],
            vec![],
        ),
    }
}
